using Pkg; Pkg.activate(".")
using DifferentialEquations, SciMLSensitivity, Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays, Plots, Random, Statistics, Printf, ForwardDiff

include("../src/tools.jl")
include("../src/models.jl")
include("../src/physics.jl")


# This is the schedule the model sees during training
# Regular doses at Day 10, 20, 30.
# The Pure AI will likely memorize these specific timestamps

function protocol_train(t)
    dose = 0.0
    for t_d in [10.0, 20.0, 30.0] 
        if t >= t_d
            dose += 1.0 * exp(-0.25 * (t - t_d))
        end
    end
    return dose
end


# This represents a completely different patient or a changed prescription
# Doses are irregular : Day 15, 45, 80
# The model has never seen these numbers

function protocol_new(t)
    dose = 0.0
    for t_d in [15.0, 45.0, 80.0]
        if t >= t_d
            dose += 1.0 * exp(-0.25 * (t - t_d))
        end
    end
    return dose
end


u0 = [0.1]
t_train_span = (0.0, 60.0)
t_data = 0.0:0.5:60.0

# This is the only reality the models are allowed to see.
prob_train = ODEProblem((u,p,t)->ground_truth_dynamics(u,p,t,protocol_train), u0, t_train_span)
y_train_true = Array(solve(prob_train, Vern7(), saveat=t_data))

rng = Random.MersenneTwister(42)
y_train_noisy = y_train_true .* (1.0 .+ 0.03 .* randn(rng, size(y_train_true)))


rng_model = Random.MersenneTwister(123)
nn, p_init, st = get_optimized_ude_model(rng_model)


# dN/dt = Gompertz - NN(N, C)
function ude_dyn(du, u, p, t, proto_func)
    N = max(u[1], 1e-6)
    phys = ude_known_physics(N, 0.50) 
    
    # The NN sees the concentration proto_func(t)
    # It learns "If Concentration is high -> Reduce Growth"
    nn_out = first(nn([N, proto_func(t)], p, st))[1] 
    correction = 3.0 * ((nn_out + 1.0)/2.0) * N
    
    du[1] = phys - correction
end


# dN/dt = NN(N, C)
# It has to learn everything from scratch.

function pure_dyn(du, u, p, t, proto_func)
    N = u[1]
    nn_out = first(nn([N, proto_func(t)], p, st))[1]
    du[1] = 5.0 * nn_out 
end


# Generic training function to keep things clean

function train(dyn_func, label)
    #  We pass protocol_train here
    prob = ODEProblem((du,u,p,t)->dyn_func(du,u,p,t,protocol_train), u0, t_train_span, p_init)

    loss(p, _) = mean(abs2, log.(Array(solve(...)) .+ 1e-9) .- log.(y_train_noisy .+ 1e-9))    
    optf = Optimization.OptimizationFunction(loss, Optimization.AutoForwardDiff())
    
    # Adam -> BFGS
    res1 = Optimization.solve(Optimization.OptimizationProblem(optf, p_init), OptimizationOptimisers.Adam(0.05), maxiters=300)
    res2 = Optimization.solve(Optimization.OptimizationProblem(optf, res1.u), OptimizationOptimJL.LBFGS(), maxiters=100)
    
    println("Done (Loss: $(round(res2.objective, digits=5)))")
    return res2.u
end

p_ude = train(ude_dyn, "Hybrid UDE")
p_pure = train(pure_dyn, "Pure AI")

t_test_span = (0.0, 100.0)
t_test = 0.0:0.5:100.0

prob_test_true = ODEProblem((u,p,t)->ground_truth_dynamics(u,p,t,protocol_new), u0, t_test_span)
y_test_true = Array(solve(prob_test_true, Vern7(), saveat=t_test))

# We do not retrain the models We use the weights p_ude and p_pure we just learned
# We simply inject protocol_new into the ODE solver

# Hybrid UDE on New Protocol
prob_ude_new = ODEProblem((du,u,p,t)->ude_dyn(du,u,p,t,protocol_new), u0, t_test_span, p_ude)
y_ude_new = Array(solve(prob_ude_new, Vern7(), saveat=t_test))

# Pure AI on New Protocol
prob_pure_new = ODEProblem((du,u,p,t)->pure_dyn(du,u,p,t,protocol_new), u0, t_test_span, p_pure)
y_pure_new = Array(solve(prob_pure_new, Vern7(), saveat=t_test))



# Metric Calculation
mse_ude = mean(abs2, y_ude_new .- y_test_true)
mse_pure = mean(abs2, clamp.(y_pure_new, 0, 2.0) .- y_test_true) 

println("Pure AI MSE  : $mse_pure")
println("Hybrid MSE   : $mse_ude")
println("Gain         : $(round(mse_pure/mse_ude, digits=1))x")


p = plot(title="Generalization Test: New Dosing Schedule", layout=(1,1))

plot!(p, t_test, y_test_true[1,:], label="Ground Truth (New Protocol)", c=:black, lw=2, ls=:dash)

plot!(p, t_test, clamp.(y_pure_new[1,:], 0, 1.5), label="Pure AI (Fails)", c=:orange, lw=2)
plot!(p, t_test, y_ude_new[1,:], label="Hybrid UDE (Generalizes)", c=:green, lw=2)

vline!(p, [15.0, 45.0, 80.0], c=:red, ls=:dot, label="New Doses (Unseen)", alpha=0.5)

ylabel!(p, "Tumor Volume")
xlabel!(p, "Time (Days)")

savefig("docs/generalization.png")
