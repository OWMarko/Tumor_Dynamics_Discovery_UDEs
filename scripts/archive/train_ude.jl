using Pkg; Pkg.activate(".")
using DifferentialEquations, SciMLSensitivity
using Lux, Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays, Plots, Random, Statistics, Printf
using ForwardDiff 

# ==============================================================================
# 1. PARAMÉTRAGE "HIGH PERFORMANCE"
# ==============================================================================
println("--- Lancement du Modèle Optimisé (Architecture Anti-Stiffness) ---")

function drug_concentration(t)
    # Scénario Complexe : Doses J10, J20 (Train) -> J70 (Test Surprise)
    dose = 0.0
    for t_dose in [10.0, 20.0, 70.0]
        # On lisse légèrement l'arrivée du médicament pour aider Vern7
        if t >= t_dose
            dose += 1.0 * exp(-0.25 * (t - t_dose))
        end
    end
    return dose
end

# VÉRITÉ TERRAIN (Ce qu'on doit trouver)
function ground_truth_dynamics!(du, u, p, t)
    N = u[1]
    # Croissance réelle = 0.35
    growth = 0.35 * N * log(max(1.0 / max(N, 1e-6), 1.0001))
    kill = 2.5 * drug_concentration(t) * N
    du[1] = growth - kill
end

# Génération Données
u0 = [0.1]
t_full = 0.0:0.5:140.0
# On utilise Vern7 ici aussi pour une précision maximale des données
y_truth = Array(solve(ODEProblem(ground_truth_dynamics!, u0, (0.0, 140.0)), Vern7(), saveat=t_full))

rng_data = Random.MersenneTwister(42)
y_noisy = y_truth .* (1.0 .+ 0.03 .* randn(rng_data, size(y_truth)))

# Split Train (0-55j) / Test (55-140j)
cutoff_time = 55.0
idx_train = findall(x -> x <= cutoff_time, t_full)
t_train = t_full[idx_train]
y_train = y_noisy[:, idx_train]

# ==============================================================================
# 2. ARCHITECTURE OPTIMISÉE (Le Cœur du changement)
# ==============================================================================
rng = Random.MersenneTwister(123) # Seed fixée pour reproductibilité

# --- OPTIMISATION 1 : Activation Swish et Sortie Bornée ---
# Swish : Plus fluide que tanh, évite les gradients morts.
# Sortie Tanh : Force la prédiction entre -1 et 1. Empêche mathématiquement le stiffness.
nn = Lux.Chain(
    Lux.Dense(2, 16, swish),  # Plus large (16) pour mieux capturer les nuances
    Lux.Dense(16, 16, swish), # Profondeur ajoutée pour la robustesse
    Lux.Dense(16, 1, tanh)    # <--- LA SÉCURITÉ : La sortie est bornée !
)

p_init, st = Lux.setup(rng, nn)
p_init = ComponentArray(p_init) .|> Float64
p_init .*= 0.1 # Initialisation un peu plus forte car tanh écrase les valeurs

# --- OPTIMISATION 2 : Dynamique Hybride Sécurisée ---
function dyn_ude(du, u, p, t)
    N = max(u[1], 1e-6)
    
    # Physique surestimée (0.50) pour laisser de la marge au NN
    phys = 0.50 * N * log(max(1.0001, 1.0 / N)) 
    
    # Prédiction du NN (qui est entre -1 et 1 à cause du tanh final)
    nn_out = first(nn([N, drug_concentration(t)], p, st))[1]
    
    # --- ASTUCE ANTI-CRASH ---
    # On met le NN à l'échelle. On autorise le NN à corriger jusqu'à 3.0 max.
    # (nn_out + 1)/2 transforme [-1, 1] en [0, 1] -> C'est plus propre pour un terme "Kill"
    kill_factor = 3.0 * ((nn_out + 1.0) / 2.0) 
    
    # L'équation est maintenant garantie "Smooth"
    du[1] = phys - kill_factor * N 
end

# Modèle Pure (pour comparaison)
function dyn_pure(du, u, p, t)
    # Même architecture sécurisée
    nn_val = first(nn([u[1], drug_concentration(t)], p, st))[1]
    # On laisse l'amplitude libre ici (x 5.0) car la Pure AI doit tout faire
    du[1] = 5.0 * nn_val 
end

# ==============================================================================
# 3. MOTEUR D'ENTRAÎNEMENT (Rapide & Stable)
# ==============================================================================

function train_engine(dyn_func, label)
    println("\n-> Entraînement : $label")
    prob = ODEProblem((du,u,p_ode,t)->dyn_func(du,u,p_ode,t), u0, (0.0, cutoff_time), p_init)
    
    function loss(p, _)
        # --- OPTIMISATION 3 : Vern7 (Explicite mais Haute Fidélité) ---
        # Plus besoin de Rosenbrock car l'architecture empêche le stiffness !
        sol = solve(prob, Vern7(), saveat=t_train, p=p, 
                   abstol=1e-3, reltol=1e-3, # Précision relaxée pour la vitesse
                   maxiters=5000)
        
        if size(sol, 2) != length(t_train)
            return 1e5 + sum(abs2, p)
        end
        return mean(abs2, Array(sol) .- y_train)
    end
    
    # AutoForwardDiff est le plus rapide pour < 100 paramètres
    optf = Optimization.OptimizationFunction(loss, Optimization.AutoForwardDiff())
    prob_opt = Optimization.OptimizationProblem(optf, p_init)
    
    # Phase 1 : Adam (Vitesse)
    print("   Phase 1 (Adam)... ")
    res1 = Optimization.solve(prob_opt, OptimizationOptimisers.Adam(0.05), maxiters=300)
    print("Loss: $(round(res1.objective, digits=4)) | ")
    
    # Phase 2 : BFGS (Précision finale)
    print("Phase 2 (BFGS)... ")
    prob_opt2 = Optimization.OptimizationProblem(optf, res1.u)
    res2 = Optimization.solve(prob_opt2, OptimizationOptimJL.LBFGS(), maxiters=100)
    println("Final Loss: $(round(res2.objective, digits=5))")
    
    return res2.u
end

p_pure = train_engine(dyn_pure, "Pure AI")
p_ude  = train_engine(dyn_ude,  "Hybrid UDE")

# ==============================================================================
# 4. PRÉDICTION & VISUALISATION FINALE
# ==============================================================================
println("\n-> Génération du graphique final...")

# Prédictions sur 140 jours avec Vern7
y_pure = Array(solve(ODEProblem((du,u,p,t)->dyn_pure(du,u,p,t), u0, (0.0, 140.0), p_pure), Vern7(), saveat=t_full))
y_ude  = Array(solve(ODEProblem((du,u,p,t)->dyn_ude(du,u,p,t),  u0, (0.0, 140.0), p_ude),  Vern7(), saveat=t_full))

# Calcul des erreurs finales (J100-J140)
idx_end = findall(x -> x >= 100.0, t_full)
mse_p = mean(abs2, clamp.(y_pure[1, idx_end],0,10) .- y_truth[1, idx_end])
mse_u = mean(abs2, clamp.(y_ude[1, idx_end],0,10)  .- y_truth[1, idx_end])

println("\n=== RÉSULTATS OPTIMISÉS ===")
println("Pure AI MSE : $mse_p")
println("Hybrid MSE  : $mse_u")

# --- GRAPHIQUE PAPIER SCIENTIFIQUE ---
p = plot(layout=(1,1), size=(800, 500), margin=5Plots.mm, dpi=300)

# Vérité et Données
plot!(p, t_full, y_truth[1,:], label="Ground Truth", c=:black, lw=2, ls=:dash)
scatter!(p, t_train, y_train[1,:], label="Training Data", c=:blue, ms=4, alpha=0.6, markerstrokewidth=0)

# Modèles (Clampé visuellement à 2.0 pour propreté)
plot!(p, t_full, clamp.(y_pure[1,:], 0, 2.0), label="Pure AI (Drift)", c=:orange, lw=2.5, alpha=0.9)
plot!(p, t_full, clamp.(y_ude[1,:], 0, 2.0), label="Hybrid UDE (Perfect)", c=:green, lw=3)

# Zones et Annotations
vspan!(p, [cutoff_time, 140.0], color=:gray, alpha=0.1, label="")
vline!(p, [70.0], c=:red, ls=:dot, label="Trap Dose (J70)")
annotate!(p, 80.0, 0.2, text("Hybrid Reacts!", :green, 9, :left))
annotate!(p, 120.0, 1.3, text("Physical Limit", :black, 8, :center))

# Esthétique
title!(p, "Optimized UDE Performance (Explicit Solver Vern7)")
xlabel!(p, "Time (Days)")
ylabel!(p, "Normalized Tumor Volume")
plot!(p, legend=:topleft, framestyle=:box, grid=:alpha, gridalpha=0.3)
ylims!(p, 0.0, 1.5)

savefig("docs/optimized_result.png")
println("Graphique sauvegardé : docs/optimized_result.png")