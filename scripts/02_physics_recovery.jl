using Pkg; Pkg.activate(".")
using DifferentialEquations, SciMLSensitivity, Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays, Plots, Random, Statistics, ForwardDiff

# Imports des modules communs
include("../src/tools.jl")
include("../src/models.jl")
include("../src/physics.jl")

println("--- EXP 02: Physics Recovery (Multi-Peak) ---")

# 1. SCÉNARIO COMPLEXE (3 Doses pour forcer l'apprentissage de la répétition)
function drug_conc_multi(t)
    dose = 0.0
    # Doses à J10, J25, J40
    for t_d in [10.0, 25.0, 40.0]
        if t >= t_d
            # Décroissance exponentielle standard
            dose += 1.0 * exp(-0.3 * (t - t_d)) 
        end
    end
    return dose
end

# 2. GÉNÉRATION DES DONNÉES (Sur 60 jours)
u0 = [0.1]
t_span = (0.0, 60.0)
t_data = 0.0:0.5:60.0

# On génère la vérité avec les 3 pics
y_true = Array(solve(ODEProblem((u,p,t)->ground_truth_dynamics(u,p,t,drug_conc_multi), u0, t_span), Vern7(), saveat=t_data))
rng = Random.MersenneTwister(42)
y_noisy = y_true .* (1.0 .+ 0.03 .* randn(rng, size(y_true)))

println("Données générées avec 3 pics de chimio.")

# 3. PRÉPARATION DU MODÈLE UDE
# On récupère une nouvelle initialisation propre
nn, p_init, st = get_optimized_ude_model(Random.MersenneTwister(123))

function ude_dyn(du, u, p, t)
    N = max(u[1], 1e-6)
    
    # Physique CONNUE mais FAUSSE (Surestimée à 0.50 au lieu de 0.35)
    phys = ude_known_physics(N, 0.50)
    
    # Correction Neurale
    nn_out = first(nn([N, drug_conc_multi(t)], p, st))[1]
    
    # Mise à l'échelle pour éviter le stiffness
    correction = 3.0 * ((nn_out + 1.0)/2.0) * N
    
    du[1] = phys - correction
end

# 4. ENTRAÎNEMENT
prob = ODEProblem(ude_dyn, u0, t_span, p_init)

function loss(p, _)
    sol = solve(prob, Vern7(), saveat=t_data, p=p, abstol=1e-3, reltol=1e-3)
    if size(sol, 2) != length(t_data)
        return 1e5
    end
    return mean(abs2, Array(sol) .- y_noisy)
end

optf = Optimization.OptimizationFunction(loss, Optimization.AutoForwardDiff())
prob_opt = Optimization.OptimizationProblem(optf, p_init)

println("-> Phase 1: ADAM (Exploration)...")
res1 = Optimization.solve(prob_opt, OptimizationOptimisers.Adam(0.05), maxiters=500)

println("-> Phase 2: BFGS (Précision)...")
# On utilise les résultats d'ADAM comme point de départ
prob_opt2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(prob_opt2, OptimizationOptimJL.LBFGS(), maxiters=200)

println("-> Entraînement terminé. Analyse de la loi cachée...")

# 5. RECONSTRUCTION DE LA LOI PHYSIQUE
# On va comparer terme à terme ce que le réseau a trouvé vs la réalité
true_missing = Float64[]
learned_term = Float64[]
t_fine = 0.0:0.1:60.0

for t in t_fine
    # On fixe N à une valeur constante pour isoler l'effet temporel du médicament
    # N=0.5 est un bon point de fonctionnement (milieu de croissance)
    N_fixe = 0.5 
    
    # A. Ce que le réseau prédit (Le terme correctif)
    nn_out = first(nn([N_fixe, drug_conc_multi(t)], res2.u, st))[1]
    valeur_NN = (3.0 * ((nn_out + 1.0)/2.0) * N_fixe)
    push!(learned_term, valeur_NN)
    
    # B. Ce qu'il DEVAIT trouver (La différence mathématique exacte)
    # Missing = Physique_Fausse - Vraie_Dynamique
    # Missing = (0.50 * ...) - [(0.35 * ...) - (Kill * ...)]
    # Missing = (0.15 * ...) + (Kill * ...)
    
    phys_fausse = ude_known_physics(N_fixe, 0.50)
    # Note: ground_truth_dynamics renvoie maintenant un vecteur [val], on prend [1]
    vraie_dyn = ground_truth_dynamics([N_fixe], nothing, t, drug_conc_multi)[1]
    
    push!(true_missing, phys_fausse - vraie_dyn)
end

# 6. GRAPHIQUE FINAL
p = plot(t_fine, true_missing, label="Theoretical Missing Term (Ground Truth)", 
         c=:black, ls=:dash, lw=2, title="Fig 4: Recovering Complex Dynamics")

plot!(p, t_fine, learned_term, label="Neural Network Reconstruction", 
      c=:red, lw=2, alpha=0.8)

# Esthétique
xlabel!(p, "Time (Days)")
ylabel!(p, "Growth Correction + Drug Effect")
ylims!(p, 0.0, maximum(true_missing)*1.2) # Zoom auto
annotate!(p, 10, maximum(true_missing), text("Dose 1", 8, :center))
annotate!(p, 25, maximum(true_missing), text("Dose 2", 8, :center))
annotate!(p, 40, maximum(true_missing), text("Dose 3", 8, :center))

savefig("docs/fig4_hidden_law.png")
println("Sauvegardé : docs/fig4_hidden_law.png")