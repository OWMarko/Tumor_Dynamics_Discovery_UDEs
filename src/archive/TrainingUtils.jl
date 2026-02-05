module TrainingUtils

using DifferentialEquations, SciMLSensitivity
using Random, Statistics, ComponentArrays

"""
    generate_synthetic_data(ode_func, p_true, u0, tspan, saveat; noise_level=0.05)

Runs the ground truth simulation and adds Gaussian noise to simulate clinical sparsity.
"""
function generate_synthetic_data(ode_func, p_true, u0, tspan, saveat; noise_level=0.05)
    # 1. Solve the "Perfect" Physics
    prob = ODEProblem(ode_func, u0, tspan, p_true)
    sol = solve(prob, Tsit5(), saveat=saveat)
    
    # 2. Add Measurement Noise
    # We use multiplicative noise: y_obs = y_true * (1 + N(0, sigma))
    data = sol.u
    noisy_data = [u .* (1 .+ noise_level * randn(length(u))) for u in data]
    
    return sol.t, stack(noisy_data) # Returns time and matrix of observations
end

"""
    predict_ude(u0, p_nn, st_nn, model, t_obs, p_fixed)

Solves the Neural ODE using the current weights `p_nn`.
Note: Uses 'Rosenbrock23' if the system becomes stiff, or 'AutoTsit5' generally.
"""
# In src/TrainingUtils.jl

function predict_ude(u0, p_nn, st_nn, model, t_obs, drug_func)
    
    function ude_dynamics!(du, u, p, t)
        N = u[1]
        C_t = drug_func(t)
        nn_input = [N, C_t]
        du[1] = first(model(nn_input, p, st_nn))[1]
    end

    prob = ODEProblem(ude_dynamics!, u0, (t_obs[1], t_obs[end]), p_nn)
    
    # --- MODIFICATION HERE ---
    # We replaced AutoTsit5(Rosenbrock23()) with just Rosenbrock23().
    # We also verify the sensealg is explicitly set to InterpolatingAdjoint() which is stable.
    return solve(prob, Rosenbrock23(), saveat=t_obs, sensealg=InterpolatingAdjoint())
end

export generate_synthetic_data, predict_ude

end # module