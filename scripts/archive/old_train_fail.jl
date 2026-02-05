using Pkg; Pkg.activate(".")
using DifferentialEquations, SciMLSensitivity
using Lux, Optimization, OptimizationOptimisers
using ComponentArrays, Random, Statistics, Zygote 

include("../src/TumorModels.jl")
using .TumorModels

println("=== BENCHMARK: PRIME / NAIVE APPROACH (IMPLICIT SOLVER) ===")
println("Configuration: Rosenbrock23 (Stiff Solver) + Adjoint.")
println("Note: We run ONLY 5 epochs to estimate the total time without waiting hours.")

# ==============================================================================
# 1. DATA GENERATION (GOMPERTZ)
# ==============================================================================
function ground_truth_dynamics!(du, u, p, t)
    N = max(u[1], 1e-5)
    growth = p.rho * N * log(p.K / N)
    drug = drug_concentration(t)
    kill = p.delta * drug * N
    du[1] = growth - kill
end

p_true = (rho=0.25, K=1.0, delta=1.5)
u0 = [0.1]
tspan = (0.0, 60.0)
saveat = 2.0 

# On génère les données rapidement (la vérité terrain n'a pas besoin d'être lente)
prob_true = ODEProblem(ground_truth_dynamics!, u0, tspan, p_true)
sol_true = solve(prob_true, Tsit5(), saveat=saveat)

rng_data = Random.default_rng()
y_obs = Array(sol_true) + 0.05 * Array(sol_true) .* randn(rng_data, size(Array(sol_true)))
t_obs = sol_true.t
y_obs = Float64.(y_obs)
t_obs = Float64.(t_obs)
u0 = [y_obs[1]]

println("Data loaded: $(length(t_obs)) points.")

# ==============================================================================
# 2. NEURAL NETWORK
# ==============================================================================
nn_model = create_pure_ude() 
p_init, st = Lux.setup(Random.default_rng(), nn_model)
p_nn = ComponentArray(p_init) .|> Float64
p_nn .*= 0.01

# ==============================================================================
# 3. SLOW PREDICTION (ROSENBROCK23)
# ==============================================================================
function predict_implicit(p)
    function ude_dynamics!(du, u, p, t)
        N = max(u[1], 1e-5)
        C_t = drug_concentration(t)
        nn_out = first(nn_model([N, C_t], p, st))[1]
        
        rho_est = 0.25; K_est = 1.0
        physics = rho_est * N * log(K_est / N)
        du[1] = physics - abs(nn_out)
    end
    
    prob = ODEProblem(ude_dynamics!, u0, (t_obs[1], t_obs[end]), p)
    
    # --- LE FREIN À MAIN ---
    # Rosenbrock23() : Solveur implicite (Stiff).
    # Il doit inverser la Jacobienne à chaque pas.
    # C'est ce qui tue la performance sur le CPU.
    return solve(prob, Rosenbrock23(), saveat=t_obs, 
                 sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
end

function loss_implicit(p, nothing)
    pred_sol = predict_implicit(p)
    if pred_sol.retcode != ReturnCode.Success; return Inf; end
    y_pred = Array(pred_sol)
    mse = mean(abs2, y_pred .- y_obs)
    return mse
end

# ==============================================================================
# 4. BENCHMARK RUN (5 EPOCHS ONLY)
# ==============================================================================
println("--- Starting Benchmark (5 Iterations) ---")

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction(loss_implicit, adtype)
optprob = Optimization.OptimizationProblem(optf, p_nn)

println("Running optimizer with Rosenbrock23 (Implicit)...")
# On chronomètre 5 époques
@time res_bench = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.005), maxiters=5)

println("--- Benchmark Complete ---")
println("CALCULATION FOR REPORT:")
println("Multiply the time displayed above by 20 to get the estimate for 100 epochs.")
println("Example: if 5 epochs = 80s, then 100 epochs ≈ 1600s (26 mins).")