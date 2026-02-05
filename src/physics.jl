# This is the False Physics we feed into the Neural UDE.
# The goal is to trick the model we tell it the tumor grows very fast (rate = 0.50),
# whereas in reality it grows slower (0.35)
# The AI will have to fight against this wrong assumption and learn to correct it.

function ude_known_physics(N, p_growth=0.50)
    
    # We use max here as a safety net. 
    # Logarithms explode if N reaches 0 so we clamp the value to ensure numerical stability
    
    return p_growth * N * log(max(1.0001, 1.0 / max(N, 1e-6)))
end

# This function represents the Real World (or the patient)
# We use this only to generate the training data. The AI never sees this code.

function ground_truth_dynamics(u, p, t, dose_func)
    N = u[1]
    
    # The natural biology of the tumor (Gompertz law)
    growth = 0.35 * N * log(max(1.0 / max(N, 1e-6), 1.0001))
    
    # The therapy effect : Drug Concentration * Potency * Tumor Size
    kill = 2.5 * dose_func(t) * N
    

    # DifferentialEquations.jl expects a Vector as output not a single number
    return [growth - kill] 
end
