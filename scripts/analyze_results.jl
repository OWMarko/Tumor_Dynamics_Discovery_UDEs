using Pkg; Pkg.activate(".")
using DifferentialEquations, SciMLSensitivity
using Lux, Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays, Plots, Random, Statistics
using FiniteDiff 
using Zygote

# ==============================================================================
# 1. OUTILS MATHÉMATIQUES (Stable Mode)
# ==============================================================================

# Lissage de abs() pour dériver proprement
function smooth_abs(x)
    return sqrt(x^2 + 1e-6)
end

function drug_concentration(t)
    dose = 0.0
    for t_dose in [10.0, 20.0, 30.0]
        if t >= t_dose
            dose += 1.0 * exp(-0.35 * (t - t_dose))
        end
    end
    return dose
end

# ==============================================================================
# 2. GÉNÉRATION DES DONNÉES
# ==============================================================================
println("\n--- 1. Generating Clinical Data ---")

function ground_truth_dynamics!(du, u, p, t)
    N = max(u[1], 1e-5)
    # Protection log pour éviter NaN
    growth = p.rho * N * log(max(p.K / N, 1.0001)) 
    kill = p.delta * drug_concentration(t) * N
    du[1] = growth - kill
end

p_true = (rho=0.25, K=1.0, delta=1.5)
u0 = [0.1]
t_data = 0.0:2.0:60.0 # Données éparses

prob_true = ODEProblem(ground_truth_dynamics!, u0, (0.0, 90.0), p_true)
sol_true = solve(prob_true, Tsit5(), saveat=t_data)
y_true = Array(sol_true)

rng = Random.default_rng()
y_noisy = y_true .* (1.0 .+ 0.05 .* randn(rng, size(y_true)))

cutoff_time = 40.0
idx_train = findall(x -> x <= cutoff_time, t_data)
t_train = t_data[idx_train]
y_train = y_noisy[:, idx_train]

println("Training Data: $(length(t_train)) points (Day 0 to 40)")

# ==============================================================================
# 3. MODÈLES & INITIALISATION (CORRECTION ICI)
# ==============================================================================

# On réduit la taille à 8 neurones pour la stabilité
nn = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 1))
p_init, st = Lux.setup(rng, nn)
p_init = ComponentArray(p_init) .|> Float64

# --- CORRECTION CRUCIALE : Initialisation Petite ---
# Cela force le réseau à ne pas perturber la physique au début.
# L'optimiseur partira d'une solution stable (Gompertz pur) et ajoutera la drogue peu à peu.
p_init .*= 0.01 

# --- A. PURE AI ---
function dynamics_pure!(du, u, p, t, model, st_nn)
    N = u[1]
    C = drug_concentration(t)
    du[1] = first(model([N, C], p, st_nn))[1]
end

# --- B. HYBRID UDE ---
function dynamics_ude!(du, u, p, t, model, st_nn)
    N = max(u[1], 1e-5)
    C = drug_concentration(t)
    
    # Physique (Gompertz)
    log_term = log(max(1e-5, 1.0 / N)) 
    physics = 0.25 * N * log_term
    
    # Correction Réseau
    nn_out = first(model([N, C], p, st_nn))[1]
    
    # On soustrait l'effet du médicament
    du[1] = physics - smooth_abs(nn_out)
end

# ==============================================================================
# 4. ENTRAÎNEMENT ROBUSTE
# ==============================================================================

function train_model(mode_name, dynamics_func, epochs)
    println("\n-> Training $mode_name...")
    
    function prob_func(p)
        ODEProblem((du,u,p_ode,t) -> dynamics_func(du,u,p,t,nn,st), u0, (0.0, cutoff_time), p)
    end

    function loss(p, nothing)
        prob = prob_func(p)
        # On augmente maxiters pour éviter le Loss=1e6
        sol = solve(prob, Tsit5(), saveat=t_train, abstol=1e-3, reltol=1e-3, maxiters=5000)
        
        if sol.retcode != :Success
            return 1e6 
        end
        
        pred = Array(sol)
        if size(pred) != size(y_train)
            return 1e6
        end
        return mean(abs2, pred .- y_train)
    end

    opt_func = Optimization.OptimizationFunction(loss, Optimization.AutoFiniteDiff())
    opt_prob = Optimization.OptimizationProblem(opt_func, p_init)
    
    # 500 itérations pour être sûr de converger
    res = Optimization.solve(opt_prob, OptimizationOptimisers.Adam(0.05), maxiters=epochs)
    println("   Final Loss: $(res.objective)")
    return res.u
end

# On augmente un peu les époques car on part de 0.01
p_pure_trained = train_model("Pure AI", dynamics_pure!, 500)
p_ude_trained  = train_model("Hybrid UDE", dynamics_ude!, 500)

# ==============================================================================
# 5. PRÉDICTION & GRAPHIQUE
# ==============================================================================
println("\n-> Running Forecast Comparison...")

t_full = 0.0:0.5:90.0
y_truth_full = Array(solve(ODEProblem(ground_truth_dynamics!, u0, (0.0, 90.0), p_true), Tsit5(), saveat=t_full))[1,:]

function predict_safe(dyn_func, p_trained)
    prob = ODEProblem((du,u,p_ode,t) -> dyn_func(du,u,p_trained,t,nn,st), u0, (0.0, 90.0), p_trained)
    # On utilise Rosenbrock23 pour la prédiction finale (graphique) au cas où ça explose
    sol = solve(prob, Rosenbrock23(), saveat=t_full, abstol=1e-2, reltol=1e-2, maxiters=5000)
    
    y = Array(sol)[1,:]
    if length(y) < length(t_full)
        y = vcat(y, fill(NaN, length(t_full) - length(y)))
    end
    return y
end

y_pure = predict_safe(dynamics_pure!, p_pure_trained)
y_ude  = predict_safe(dynamics_ude!,  p_ude_trained)

# MSE TEST (J40 -> J90)
idx_test = findall(x -> x >= cutoff_time, t_full)
mse_pure = mean(abs2, filter(!isnan, y_pure[idx_test] .- y_truth_full[idx_test]))
mse_ude  = mean(abs2, filter(!isnan, y_ude[idx_test]  .- y_truth_full[idx_test]))

println("\n=== RÉSULTATS FINAUX ===")
println("Pure AI MSE: $(round(mse_pure, digits=2))")
println("Hybrid MSE:  $(round(mse_ude, digits=2))")

# Graphique avec Clamping
y_pure_plot = clamp.(y_pure, 0.0, 2.5) 
y_ude_plot  = clamp.(y_ude, 0.0, 2.5)

p = plot(t_full, y_truth_full, label="Truth", color=:black, lw=2, ls=:dash)
plot!(p, t_full, y_pure_plot, label="Pure AI (Unstable)", color=:orange, lw=3)
plot!(p, t_full, y_ude_plot, label="Hybrid UDE (Stable)", color=:green, lw=3)

vspan!(p, [cutoff_time, 90.0], color=:gray, alpha=0.1, label="Forecast Zone")
vline!(p, [cutoff_time], color=:blue, ls=:dot)
scatter!(p, t_train, y_train[1,:], label="Training Data", color=:blue, ms=4)

title!(p, "Benchmark: Pure AI vs Hybrid UDE")
yaxis!(p, (0, 2.0))

savefig("docs/benchmark_final_fixed.png")
println("Graphique sauvegardé.")