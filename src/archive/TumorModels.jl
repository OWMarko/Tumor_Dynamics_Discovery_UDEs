module TumorModels

using Lux, Random, ComponentArrays
using DifferentialEquations

# This code ..

function drug_concentration(t)
    dose = 0.0
    k_elim = 0.35  # Elimination rate
    for t_dose in [10.0, 20.0, 30.0]
        if t >= t_dose
            dose += 1.0 * exp(-k_elim * (t - t_dose))
        end
    end
    return dose
end

function ground_truth_dynamics!(du, u, p, t)
    N = u[1]
    ρ, K, δ = p.rho, p.K, p.delta
    C_t = drug_concentration(t)
    
    # Logistic Growth - Drug Kill
    du[1] = ρ * N * (1 - N/K) - δ * C_t * N
end

# ==============================================================================
# 2. UNIVERSAL DIFFERENTIAL EQUATION (The Student)
# ==============================================================================

"""
    create_ude_architecture()

Returns a Lux Neural Network designed to approximate the unknown growth laws.
Input:  [N(t), C(t)] (Tumor size and Drug conc)
Output: [dN/dt]      (Rate of change)
"""
# ... (Gardez le début du fichier TumorModels.jl identique)

# ==============================================================================
# 2. HYBRID ARCHITECTURES (The Gray-Box)
# ==============================================================================

"""
    create_pure_ude()
Approach 1: The NN learns the ENTIRE derivative f(u, p, t).
"""
function create_pure_ude()
    return Chain(
        Dense(2, 64, tanh),
        Dense(64, 64, tanh),
        Dense(64, 1)
    )
end

"""
    hybrid_dynamics!(du, u, p, t, nn_model, p_nn, st_nn, drug_func)
Approach 2: Gray-Box.
Physics is known (Logistic Growth), NN learns ONLY the Drug Interaction term.
du/dt = rho*N*(1-N/K) - NN(N, C)
"""
function hybrid_dynamics!(du, u, p, t, nn_model, p_nn, st_nn, drug_func)
    N = u[1]
    C_t = drug_func(t)
    
    # 1. Known Physics (e.g., estimated rho=0.2 from prior knowledge)
    # We assume we know the tumor grows, but we don't know how the drug kills it.
    rho_est = 0.25 
    K_est = 1.0
    physics_term = rho_est * N * (1 - N/K_est)
    
    # 2. Unknown Interaction (Neural Network)
    # NN takes [N, C] and outputs the "Kill Rate"
    nn_out = first(nn_model([N, C_t], p_nn, st_nn))[1]
    
    # Combine: Physics - Learned_Treatment
    du[1] = physics_term - abs(nn_out) # Force kill term to be negative/positive logic
end

export drug_concentration, ground_truth_dynamics!, create_pure_ude, hybrid_dynamics!
# ... (Fin du module)
export drug_concentration, ground_truth_dynamics!, create_ude_architecture

end # module