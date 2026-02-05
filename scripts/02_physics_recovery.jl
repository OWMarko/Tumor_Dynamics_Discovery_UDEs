using Pkg; Pkg.activate(".")
using DifferentialEquations, SciMLSensitivity, Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays, Plots, Random, Statistics, ForwardDiff

include("../src/tools.jl")
include("../src/models.jl")
include("../src/physics.jl")

# Instead of a single pulse we have 3 distinct doses
# This forces the network to learn a repeatable law
# Doses at : Day 10, Day 25, Day 40.

function drug_conc_multi(t)
    dose = 0.0
    for t_d in [10.0, 25.0, 40.0]
        if t >= t_d
            dose += 1.0 * exp(-0.3 * (t - t_d)) 
        end
    end
    return dose
end

u0 = [0.1]
t_span = (0.0, 60.0)
t_data = 0.0:0.5:60.0

# We use the ground_truth function from physics.jl (Growth 0.35, Kill 2.5)
y_true = Array(solve(ODEProblem((u,p,t)->ground_truth_dynamics(u,p,t,drug_conc_multi), u0, t_span), Vern7(), saveat=t_data))

# Add noise to make it realistic
rng = Random.MersenneTwister(42)
y_noisy = y_true .* (1.0 .+ 0.03 .* randn(rng, size(y_true)))

println("Données générées avec 3 pics de chimio.")

# We initialize the neural network with small weights
nn, p_init, st = get_optimized_ude_model(Random.MersenneTwister(123))

function ude_dyn(du, u, p, t)
    N = max(u[1], 1e-6)
    
    # We tell it the growth rate is 0.50 (instead of 0.35)
    # This creates 
    # The Growth Error (0.50 - 0.35 = 0.15 * N * log)
    # The Drug Effect (which is totally missing from the prior)
    phys = ude_known_physics(N, 0.50)
    
    # The Neural Network must learn to fill this gap
    nn_out = first(nn([N, drug_conc_multi(t)], p, st))[1]
    
    # Scaling to help the optimizer
    correction = 3.0 * ((nn_out + 1.0)/2.0) * N
    
    du[1] = phys - correction
end

prob = ODEProblem(ude_dyn, u0, t_span, p_init)

function loss(p, _)
    
    # We solve the UDE and compare it to the noisy data
    
    sol = solve(prob, Vern7(), saveat=t_data, p=p, abstol=1e-3, reltol=1e-3)
    if size(sol, 2) != length(t_data)
        return 1e5 # Penalty for divergence
    end
    return mean(abs2, Array(sol) .- y_noisy)
end

optf = Optimization.OptimizationFunction(loss, Optimization.AutoForwardDiff())
prob_opt = Optimization.OptimizationProblem(optf, p_init)

println("-> Phase 1 : ADAM (Exploration)...")
res1 = Optimization.solve(prob_opt, OptimizationOptimisers.Adam(0.05), maxiters=500)

println("-> Phase 2 : BFGS (Précision)...")
prob_opt2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(prob_opt2, OptimizationOptimJL.LBFGS(), maxiters=200)

# We check if the NN actually learned the physics or if it just memorized dots.

true_missing = Float64[]
learned_term = Float64[]
t_fine = 0.0:0.1:60.0


for t in t_fine
    # We fix N to a constant value to isolate the time dep effect of the drug
    # This effectively takes a slice of the learned function
    N_fixe = 0.5 
    
    # What the Neural Network learned 
    nn_out = first(nn([N_fixe, drug_conc_multi(t)], res2.u, st))[1]
    valeur_NN = (3.0 * ((nn_out + 1.0)/2.0) * N_fixe)
    push!(learned_term, valeur_NN)
    

    # The network should have learned (False Physics) - (Real Physics) = (0.50 * Gompertz) - (0.35 * Gompertz - KillTerm) = 0.15 * Gompertz + KillTerm
    phys_fausse = ude_known_physics(N_fixe, 0.50)
    vraie_dyn = ground_truth_dynamics([N_fixe], nothing, t, drug_conc_multi)[1]
    
    push!(true_missing, phys_fausse - vraie_dyn)
end

p = plot(t_fine, true_missing, label="Theoretical Missing Term (Ground Truth)", 
         c=:black, ls=:dash, lw=2, title="Recovering Complex Dynamics")

plot!(p, t_fine, learned_term, label="Neural Network Reconstruction", 
      c=:red, lw=2, alpha=0.8)

xlabel!(p, "Time (Days)")
ylabel!(p, "Growth Correction + Drug Effect")
ylims!(p, 0.0, maximum(true_missing)*1.2) 

savefig("docs/hidden_law.png")
