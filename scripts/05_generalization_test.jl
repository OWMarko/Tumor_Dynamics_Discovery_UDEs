using Pkg; Pkg.activate(".")
using DifferentialEquations, SciMLSensitivity, Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays, Plots, Random, Statistics, Printf, ForwardDiff

include("../src/tools.jl")
include("../src/models.jl")
include("../src/physics.jl")

println("--- EXP 06: GENERALIZATION TEST (New Protocol) ---")

# ==============================================================================
# 1. LES DEUX PROTOCOLES
# ==============================================================================

# A. Protocole d'Entraînement (Connu)
# Doses fixes et régulières
function protocol_train(t)
    dose = 0.0
    for t_d in [10.0, 20.0, 30.0] 
        if t >= t_d
            dose += 1.0 * exp(-0.25 * (t - t_d))
        end
    end
    return dose
end

# B. Protocole de Test (INCONNU / NOUVEAU)
# Doses décalées et irrégulières (Le patient change de médecin !)
function protocol_new(t)
    dose = 0.0
    # Doses complètement différentes : J15, J45, J80
    for t_d in [15.0, 45.0, 80.0]
        if t >= t_d
            dose += 1.0 * exp(-0.25 * (t - t_d))
        end
    end
    return dose
end

# ==============================================================================
# 2. GÉNÉRATION DES DONNÉES D'ENTRAÎNEMENT
# ==============================================================================
u0 = [0.1]
t_train_span = (0.0, 60.0)
t_data = 0.0:0.5:60.0

# On génère la vérité pour l'entraînement
prob_train = ODEProblem((u,p,t)->ground_truth_dynamics(u,p,t,protocol_train), u0, t_train_span)
y_train_true = Array(solve(prob_train, Vern7(), saveat=t_data))
rng = Random.MersenneTwister(42)
y_train_noisy = y_train_true .* (1.0 .+ 0.03 .* randn(rng, size(y_train_true)))

println("Données d'entraînement générées (Protocole Standard).")

# ==============================================================================
# 3. ENTRAÎNEMENT (Sur le Protocole Standard)
# ==============================================================================

# On récupère les architectures
rng_model = Random.MersenneTwister(123)
nn, p_init, st = get_optimized_ude_model(rng_model)

# --- Définition UDE ---
function ude_dyn(du, u, p, t, proto_func)
    N = max(u[1], 1e-6)
    phys = ude_known_physics(N, 0.50) # Physique fausse
    nn_out = first(nn([N, proto_func(t)], p, st))[1] # Le NN voit la concentration actuelle
    correction = 3.0 * ((nn_out + 1.0)/2.0) * N
    du[1] = phys - correction
end

# --- Définition Pure AI ---
function pure_dyn(du, u, p, t, proto_func)
    N = u[1]
    nn_out = first(nn([N, proto_func(t)], p, st))[1]
    du[1] = 5.0 * nn_out 
end

# Fonction d'entraînement générique
function train(dyn_func, label)
    # Note: On entraîne AVEC protocol_train
    prob = ODEProblem((du,u,p,t)->dyn_func(du,u,p,t,protocol_train), u0, t_train_span, p_init)
    
    loss(p, _) = mean(abs2, Array(solve(prob, Vern7(), saveat=t_data, p=p, abstol=1e-2)) .- y_train_noisy)
    
    optf = Optimization.OptimizationFunction(loss, Optimization.AutoForwardDiff())
    print("Training $label... ")
    res1 = Optimization.solve(Optimization.OptimizationProblem(optf, p_init), OptimizationOptimisers.Adam(0.05), maxiters=300)
    res2 = Optimization.solve(Optimization.OptimizationProblem(optf, res1.u), OptimizationOptimJL.LBFGS(), maxiters=100)
    println("Done (Loss: $(round(res2.objective, digits=5)))")
    return res2.u
end

p_ude = train(ude_dyn, "Hybrid UDE")
p_pure = train(pure_dyn, "Pure AI")

# ==============================================================================
# 4. LE TEST ULTIME : NOUVEAU PROTOCOLE
# ==============================================================================
println("\n--- LANCEMENT DU TEST DE GÉNÉRALISATION (Protocole Inconnu) ---")

# On simule la vérité sur le NOUVEAU protocole (Jusqu'à 100 jours)
t_test_span = (0.0, 100.0)
t_test = 0.0:0.5:100.0

prob_test_true = ODEProblem((u,p,t)->ground_truth_dynamics(u,p,t,protocol_new), u0, t_test_span)
y_test_true = Array(solve(prob_test_true, Vern7(), saveat=t_test))

# PRÉDICTION DES MODÈLES (Sans réentraînement !)
# On utilise p_ude et p_pure appris sur le protocole A, mais on injecte protocol_new

# UDE sur Nouveau Protocole
prob_ude_new = ODEProblem((du,u,p,t)->ude_dyn(du,u,p,t,protocol_new), u0, t_test_span, p_ude)
y_ude_new = Array(solve(prob_ude_new, Vern7(), saveat=t_test))

# Pure AI sur Nouveau Protocole
prob_pure_new = ODEProblem((du,u,p,t)->pure_dyn(du,u,p,t,protocol_new), u0, t_test_span, p_pure)
y_pure_new = Array(solve(prob_pure_new, Vern7(), saveat=t_test))

# ==============================================================================
# 5. ANALYSE & GRAPHIQUE
# ==============================================================================

# Calcul des erreurs sur le nouveau scénario
mse_ude = mean(abs2, y_ude_new .- y_test_true)
mse_pure = mean(abs2, clamp.(y_pure_new, 0, 2.0) .- y_test_true) # Clamp pour éviter l'infini

println("\nRÉSULTATS SUR LE PROTOCOLE INCONNU :")
println("Pure AI MSE  : $mse_pure")
println("Hybrid MSE   : $mse_ude")
println("Gain         : $(round(mse_pure/mse_ude, digits=1))x")

# Graphique
p = plot(title="Generalization Test: New Dosing Schedule", layout=(1,1))

# Vérité (Nouveau Protocole)
plot!(p, t_test, y_test_true[1,:], label="Ground Truth (New Protocol)", c=:black, lw=2, ls=:dash)

# Prédictions
plot!(p, t_test, clamp.(y_pure_new[1,:], 0, 1.5), label="Pure AI (Fails)", c=:orange, lw=2)
plot!(p, t_test, y_ude_new[1,:], label="Hybrid UDE (Generalizes)", c=:green, lw=2)

# Marqueurs des nouvelles doses (pour montrer qu'elles ont changé)
vline!(p, [15.0, 45.0, 80.0], c=:red, ls=:dot, label="New Doses (Unseen)", alpha=0.5)

ylabel!(p, "Tumor Volume")
xlabel!(p, "Time (Days)")

savefig("docs/exp06_generalization.png")
println("Graphique sauvegardé : docs/exp06_generalization.png")