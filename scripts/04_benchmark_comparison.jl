using Pkg; Pkg.activate(".")
using DifferentialEquations, SciMLSensitivity, Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays, Plots, Random, Statistics, Printf, ForwardDiff

# On charge ton environnement propre
include("../src/tools.jl")
include("../src/models.jl")
include("../src/physics.jl")

println("--- BENCHMARK: Pure AI vs Hybrid UDE (3 Runs - Trap Scenario) ---")

# ==============================================================================
# 1. SCÉNARIO "TRAP" (Identique pour tous les runs)
# ==============================================================================
function drug_conc_trap(t)
    dose = 0.0
    # Doses à J10, J20 (Training) et J70 (Piège Extrapolation)
    for t_d in [10.0, 20.0, 70.0]
        if t >= t_d
            dose += 1.0 * exp(-0.25 * (t - t_d))
        end
    end
    return dose
end

# Configuration temporelle
t_full = 0.0:0.5:140.0
cutoff = 55.0 # Fin de l'entraînement
idx_train = findall(x -> x <= cutoff, t_full)
idx_test = findall(x -> x > 60.0, t_full) # On évalue l'erreur sur le futur (J60-140)

# ==============================================================================
# 2. DÉFINITION DES MODÈLES
# ==============================================================================

# A. Dynamique Hybride (Ton modèle optimisé)
function ude_dyn(du, u, p, t, nn, st)
    N = max(u[1], 1e-6)
    phys = ude_known_physics(N, 0.50) # Physique fausse (0.50)
    nn_out = first(nn([N, drug_conc_trap(t)], p, st))[1]
    correction = 3.0 * ((nn_out + 1.0)/2.0) * N # Scaling
    du[1] = phys - correction
end

# B. Dynamique Pure AI (Pour comparaison)
function pure_dyn(du, u, p, t, nn, st)
    N = u[1] # Pas de physique, l'IA gère tout
    nn_out = first(nn([N, drug_conc_trap(t)], p, st))[1]
    # On laisse une amplitude plus large (x5) car l'IA doit apprendre toute la dérivée
    du[1] = 5.0 * nn_out 
end

# ==============================================================================
# 3. MOTEUR D'ENTRAÎNEMENT GÉNÉRIQUE
# ==============================================================================
function train_model(mode_name, dyn_func, u0, t_train, y_train, p_init, nn, st)
    # Fonction wrapper pour l'ODE
    prob = ODEProblem((du,u,p,t) -> dyn_func(du,u,p,t,nn,st), u0, (0.0, cutoff), p_init)

    function loss(p, _)
        # Utilisation de Vern7 (Haute Performance)
        sol = solve(prob, Vern7(), saveat=t_train, p=p, abstol=1e-3, reltol=1e-3)
        if size(sol, 2) != length(t_train)
            return 1e5
        end
        return mean(abs2, Array(sol) .- y_train)
    end

    optf = Optimization.OptimizationFunction(loss, Optimization.AutoForwardDiff())
    prob_opt = Optimization.OptimizationProblem(optf, p_init)

    # Phase 1: ADAM (Vitesse)
    res1 = Optimization.solve(prob_opt, OptimizationOptimisers.Adam(0.05), maxiters=300)
    
    # Phase 2: BFGS (Précision)
    prob_opt2 = Optimization.OptimizationProblem(optf, res1.u)
    res2 = Optimization.solve(prob_opt2, OptimizationOptimJL.LBFGS(), maxiters=100)
    
    return res2.u, res2.objective
end

# ==============================================================================
# 4. BOUCLE DE BENCHMARK (3 RUNS)
# ==============================================================================
mse_pure_history = Float64[]
mse_ude_history = Float64[]

for run in 1:3
    println("\n=== RUN $run / 3 ===")
    
    # 1. Génération des données (Bruit unique par run)
    rng_data = Random.MersenneTwister(run * 100)
    u0 = [0.1]
    
    # Vérité Terrain
    prob_true = ODEProblem((u,p,t)->ground_truth_dynamics(u,p,t,drug_conc_trap), u0, (0.0, 140.0))
    y_true = Array(solve(prob_true, Vern7(), saveat=t_full))
    y_noisy = y_true .* (1.0 .+ 0.03 .* randn(rng_data, size(y_true)))
    
    # Données d'entraînement
    t_train = t_full[idx_train]
    y_train = y_noisy[:, idx_train]

    # 2. Initialisation des Réseaux (Même seed pour départ équitable)
    rng_net = Random.MersenneTwister(42 + run)
    nn, p_init, st = get_optimized_ude_model(rng_net)

    # 3. Entraînement PURE AI
    print("  Training Pure AI... ")
    p_pure, loss_pure = train_model("Pure", pure_dyn, u0, t_train, y_train, p_init, nn, st)
    println("Done (Loss: $(round(loss_pure, digits=5)))")

    # 4. Entraînement HYBRID UDE
    print("  Training Hybrid...  ")
    p_ude, loss_ude = train_model("Hybrid", ude_dyn, u0, t_train, y_train, p_init, nn, st)
    println("Done (Loss: $(round(loss_ude, digits=5)))")

    # 5. Prédiction & Calcul MSE (Sur le FUTUR uniquement)
    sol_pure = solve(ODEProblem((du,u,p,t)->pure_dyn(du,u,p,t,nn,st), u0, (0.0, 140.0), p_pure), Vern7(), saveat=t_full)
    sol_ude  = solve(ODEProblem((du,u,p,t)->ude_dyn(du,u,p,t,nn,st),  u0, (0.0, 140.0), p_ude),  Vern7(), saveat=t_full)
    
    y_pred_pure = Array(sol_pure)
    y_pred_ude  = Array(sol_ude)

    # On clamp les valeurs de l'IA Pure pour éviter les infinis dans le calcul MSE
    mse_pure = mean(abs2, clamp.(y_pred_pure[1, idx_test], 0, 10.0) .- y_true[1, idx_test])
    mse_ude  = mean(abs2, y_pred_ude[1, idx_test] .- y_true[1, idx_test])

    push!(mse_pure_history, mse_pure)
    push!(mse_ude_history, mse_ude)

    # 6. Graphique du Run
    p = plot(title="Benchmark Run $run: Trap Forecasting", layout=(1,1))
    
    # Vérité
    plot!(p, t_full, y_true[1,:], label="Truth", c=:black, lw=2, ls=:dash)
    scatter!(p, t_train, y_train[1,:], label="Train Data", c=:blue, alpha=0.5, ms=3)
    
    # Modèles
    plot!(p, t_full, clamp.(y_pred_pure[1,:], 0, 2.0), label="Pure AI (MSE=$(round(mse_pure, digits=3)))", c=:orange, lw=2)
    plot!(p, t_full, y_pred_ude[1,:], label="Hybrid UDE (MSE=$(round(mse_ude, digits=4)))", c=:green, lw=2)
    
    # Zones
    vline!(p, [cutoff], c=:blue, ls=:dot)
    vline!(p, [70.0], c=:red, ls=:dot, label="Trap Dose")
    vspan!(p, [cutoff, 140.0], c=:gray, alpha=0.1)
    
    savefig("docs/benchmark_run_$run.png")
    println("  -> Graph saved: docs/benchmark_run_$run.png")
end

# ==============================================================================
# 5. RÉSULTATS STATISTIQUES
# ==============================================================================
mean_pure = mean(mse_pure_history)
var_pure = var(mse_pure_history)
mean_ude = mean(mse_ude_history)
var_ude = var(mse_ude_history)

println("\n" * "="^60)
println("FINAL STATISTICAL REPORT (3 RUNS)")
println("="^60)
@printf "%-15s | %-15s | %-15s\n" "Metric" "Pure AI" "Hybrid UDE"
println("-"^60)
@printf "%-15s | %.5f         | %.5f\n" "Mean MSE (Exp)" mean_pure mean_ude
@printf "%-15s | %.5f         | %.5f\n" "Variance" var_pure var_ude
@printf "%-15s | %.5f         | %.5f\n" "Std Dev" sqrt(var_pure) sqrt(var_ude)
println("-"^60)
println("Improvement Ratio (Mean): $(round(mean_pure / mean_ude, digits=1))x better")
println("="^60)