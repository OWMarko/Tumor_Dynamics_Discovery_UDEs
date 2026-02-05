using Pkg; Pkg.activate(".")
using DifferentialEquations, SciMLSensitivity
using Lux, Optimization, OptimizationOptimisers
using ComponentArrays, Random, Statistics, Zygote 
using Plots

include("../src/TumorModels.jl")
using .TumorModels

println("=== BENCHMARK: THEORETICAL METHOD (ADJOINT + ZYGOTE) ===")
println("Note: Running 100 epochs. This will take a few minutes...")

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
# 2. NEURAL NETWORK SETUP
# ==============================================================================
nn_model = create_pure_ude() 
p_init, st = Lux.setup(Random.default_rng(), nn_model)
p_nn = ComponentArray(p_init) .|> Float64
p_nn .*= 0.01

# ==============================================================================
# 3. SLOW PREDICTION FUNCTION (THEORETICAL ADJOINT)
# ==============================================================================
function predict_adjoint(p)
    function ude_dynamics!(du, u, p, t)
        N = max(u[1], 1e-5)
        C_t = drug_concentration(t)
        nn_out = first(nn_model([N, C_t], p, st))[1]
        
        # Gompertz Physics
        rho_est = 0.25; K_est = 1.0
        physics = rho_est * N * log(K_est / N)
        du[1] = physics - abs(nn_out)
    end
    
    prob = ODEProblem(ude_dynamics!, u0, (t_obs[1], t_obs[end]), p)
    
    # Méthode Lente : InterpolatingAdjoint
    return solve(prob, Tsit5(), saveat=t_obs, 
                 sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
end

function loss_adjoint(p, nothing)
    pred_sol = predict_adjoint(p)
    if pred_sol.retcode != ReturnCode.Success; return Inf; end
    y_pred = Array(pred_sol)
    mse = mean(abs2, y_pred .- y_obs)
    return mse
end

callback = function (p, l)
    println("Current Loss: $l")
    return false
end

# ==============================================================================
# 4. BENCHMARK RUN (100 EPOCHS)
# ==============================================================================
println("--- Starting Benchmark (100 Iterations) ---")
println("Method: Continuous Adjoint Analysis (Zygote)")

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction(loss_adjoint, adtype)
optprob = Optimization.OptimizationProblem(optf, p_nn)

println("Running optimizer...")
@time res_bench = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.005), callback=callback, maxiters=500)

println("--- Benchmark Complete ---")

# ==============================================================================
# 5. PLOT RESULTS
# ==============================================================================
println("Generating plot...")
p_trained = res_bench.u

# Prédiction fine pour le graphe
t_fine = range(t_obs[1], t_obs[end], length=100) |> collect .|> Float64

function predict_for_plot(p, t_vals)
    function ude_dyn!(du, u, p, t)
        N = max(u[1], 1e-5)
        C_t = drug_concentration(t)
        nn_out = first(nn_model([N, C_t], p, st))[1]
        physics = 0.25 * N * log(1.0 / N)
        du[1] = physics - abs(nn_out)
    end
    prob = ODEProblem(ude_dyn!, u0, (t_vals[1], t_vals[end]), p)
    solve(prob, Tsit5(), saveat=t_vals)
end

sol_pred = predict_for_plot(p_trained, t_fine)
y_pred_fine = Array(sol_pred)

plot(t_obs, y_obs[1,:], seriestype=:scatter, label="Clinical Data", color=:blue, title="Adjoint Method (500 Epochs)")
plot!(t_fine, y_pred_fine[1,:], label="UDE Prediction", color=:green, linewidth=2)

savefig("docs/benchmark_result.png")
println("Graph saved to docs/benchmark_result.png")