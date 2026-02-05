using Pkg; Pkg.activate(".")
using DifferentialEquations, SciMLSensitivity, Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays, Plots, Random, Statistics, ForwardDiff

include("../src/tools.jl")
include("../src/models.jl")
include("../src/physics.jl")

println("--- EXP 01: Reconstruction (Dense Data) ---")

# We define how the drug flows in the body
# Doses: Administered at t=10, 20, 30
# Decay: exp(-0.25 * t). The 0.25 is how fast the body removes the drug

function drug_conc_dense(t)
    dose = 0.0
    for t_d in [10.0, 20.0, 30.0]
        
        # We accumulate the remaining drug from each injection
        (t >= t_d) && (dose += 1.0 * exp(-0.25 * (t - t_d)))
    end
    return dose
end

# Here we simulate a perfect clinical trial with frequent measurements
u0 = [0.1] # Initial tumor size relative to carrying capacity

# We take 121 points (2 per day). This dense data makes learning easier
# and serves as a sanity check before trying sparse data
t_dense = range(0.0, 60.0, length=121)

# Solver = Vern7()
y_true = Array(solve(ODEProblem((u,p,t)->ground_truth_dynamics(u,p,t,drug_conc_dense), u0, (0.0, 60.0)), Vern7(), saveat=t_dense))

# We add 5% Gaussian noise to simulate measurement errors
y_noisy = y_true .* (1.0 .+ 0.05 .* randn(Random.MersenneTwister(42), size(y_true)))

rng = Random.MersenneTwister(123)
nn, p_init, st = get_optimized_ude_model(rng)

function ude_dyn(du, u, p, t)
    N = max(u[1], 1e-6) # log(0) avoidance
    
    # We deliberately give the model a WRONG growth rate (0.50).
    # The real rate is 0.35. The NN must learn to subtract the difference.
    phys = ude_known_physics(N, 0.50) 
    
    # Input : Tumor Size (N) and Drug Concentration (C).
    nn_out = first(nn([N, drug_conc_dense(t)], p, st))[1]
    
    # (nn_out + 1)/2 : Shifts tanh output [-1, 1] to [0, 1]. The drug only kill never heals
    #  3.0 : Amplifies the gradient signal so the optimizer wakes up
    #  N : Enforces No Tumor (N=0) = No Effect
    correction = 3.0 * ((nn_out + 1.0) / 2.0) * N 

    # smooth_abs keeps it differentiable at 0.
    du[1] = phys - smooth_abs(correction)
end

prob = ODEProblem(ude_dyn, u0, (0.0, 60.0), p_init)

function loss(p, _)
    
    # If the solver fails (diverges) we return a huge loss to punish that path
    sol = solve(prob, Vern7(), saveat=t_dense, p=p, abstol=1e-3, reltol=1e-3)
    size(sol, 2) != length(t_dense) ? 1e5 : mean(abs2, Array(sol) .- y_noisy)
end

# We use Forward-Mode AD because our parameter space is small (<100 weights)
optf = Optimization.OptimizationFunction(loss, Optimization.AutoForwardDiff())
prob_opt = Optimization.OptimizationProblem(optf, p_init)

# ADAM 0.05
println("-> Phase 1 : ADAM...")
res1 = Optimization.solve(prob_opt, OptimizationOptimisers.Adam(0.05), maxiters=500)

# BFGS
println("-> Phase 2 : BFGS...")
res2 = Optimization.solve(Optimization.OptimizationProblem(optf, res1.u), OptimizationOptimJL.LBFGS(), maxiters=200)

# We generate a curve (step 0.1) to see the smooth dynamics
y_pred = Array(solve(ODEProblem(ude_dyn, u0, (0.0, 60.0), res2.u), Vern7(), saveat=0.1))

p = plot(title="Deep Training Results")
scatter!(p, t_dense, y_noisy[1,:], label="Data (121 pts)", c=:blue, alpha=0.5)
plot!(p, 0.0:0.1:60.0, y_pred[1,:], label="UDE Fit", c=:red, lw=3)
savefig("docs/reconstruction.png")
