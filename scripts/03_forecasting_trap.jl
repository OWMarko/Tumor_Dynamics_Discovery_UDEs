using Pkg; Pkg.activate(".")
using DifferentialEquations, SciMLSensitivity, Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays, Plots, Random, Statistics, ForwardDiff

include("../src/tools.jl")
include("../src/models.jl")
include("../src/physics.jl")

# We administer drugs at Day 10 and 20 (which the model will see during training)
# we also administer a trap dose at Day 70.
# The model will never see data for Day 70. It must predict this reaction purely 
# by understanding the laws of physics it learned from the first two doses.

function drug_conc_trap(t)
    dose = 0.0
    # Doses at 10, 20 (Seen) and 70 (Unseen/Trap)
    for t_d in [10.0, 20.0, 70.0]
        if t >= t_d
            dose += 1.0 * exp(-0.25 * (t - t_d))
        end
    end
    return dose
end

u0 = [0.1]
t_full = 0.0:0.5:140.0 # We simulate 140 days

prob_true = ODEProblem((u,p,t)->ground_truth_dynamics(u,p,t,drug_conc_trap), u0, (0.0, 140.0))
y_true = Array(solve(prob_true, Vern7(), saveat=t_full))

# Add realistic noise 
rng = Random.MersenneTwister(42)
y_noisy = y_true .* (1.0 .+ 0.03 .* randn(rng, size(y_true)))


# We simulate a clinical trial that stops at Day 55.
# The model thinks the world ends at Day 55.
cutoff = 55.0
idx_train = findall(x -> x <= cutoff, t_full)
t_train = t_full[idx_train]
y_train = y_noisy[:, idx_train]


rng_model = Random.MersenneTwister(123)
nn, p_init, st = get_optimized_ude_model(rng_model)

function ude_dyn(du, u, p, t)
    N = max(u[1], 1e-6)
    
    # Biased Physics we assume the tumor grows faster (0.50) than it really does (0.35)
    # The NN has to fix this error
    phys = ude_known_physics(N, 0.50)
    
    # Note that we pass drug_conc_trap(t) here
    # Even in the future (t > 55) the model will know the dose schedule
    # but it won't know the tumor response.
    
    nn_out = first(nn([N, drug_conc_trap(t)], p, st))[1]
    correction = 3.0 * ((nn_out + 1.0)/2.0) * N
    
    du[1] = phys - correction
end

# We only optimize on (0.0, cutoff)
prob = ODEProblem(ude_dyn, u0, (0.0, cutoff), p_init)

function loss(p, _)
    sol = solve(prob, Vern7(), saveat=t_train, p=p, abstol=1e-3, reltol=1e-3)
    if size(sol, 2) != length(t_train)
        return 1e5
    end
    return mean(abs2, Array(sol) .- y_train)
end

optf = Optimization.OptimizationFunction(loss, Optimization.AutoForwardDiff())
prob_opt = Optimization.OptimizationProblem(optf, p_init)

println("-> Phase 1 : ADAM (Rough fit)...")
res1 = Optimization.solve(prob_opt, OptimizationOptimisers.Adam(0.05), maxiters=300)
println("   Loss : $(res1.objective)")

println("-> Phase 2 : BFGS (Fine tuning)...")
prob_opt2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(prob_opt2, OptimizationOptimJL.LBFGS(), maxiters=100)
println("   Final Loss : $(res2.objective)")


# Now we run the simulation all the way to Day 140 using the parameters learned from Day 0-55
prob_ude_full = ODEProblem(ude_dyn, u0, (0.0, 140.0), res2.u)
y_ude_pred = Array(solve(prob_ude_full, Vern7(), saveat=t_full))


p = plot(title="Fig 5: Forecasting Trap (UDE Performance)")

plot!(p, t_full, y_true[1,:], label="Ground Truth", c=:black, lw=2, ls=:dash)
scatter!(p, t_train, y_train[1,:], label="Training Data", c=:blue, ms=4, alpha=0.6)

plot!(p, t_full, y_ude_pred[1,:], label="Hybrid UDE Forecast", c=:green, lw=3)

# Highlighting the trap
vspan!(p, [cutoff, 140.0], color=:gray, alpha=0.1, label="Unknown Future")
vline!(p, [70.0], c=:red, ls=:dot, label="Trap Dose (Hidden)")

xlabel!(p, "Time (Days)")
ylabel!(p, "Tumor Volume")
ylims!(p, 0.0, 1.2)

savefig("docs/fig5_forecasting.png")
println("Sauvegardé : docs/fig5_forecasting.png")
