using Pkg; Pkg.activate(".")
using JLD2, ComponentArrays
using Plots

include("../src/TumorModels.jl")
include("../src/TrainingUtils.jl")

using .TumorModels
using .TrainingUtils

println("--- Generating Synthetic Clinical Trial Data ---")

# 1. Configuration
p_true = (rho=0.25, K=1.0, delta=1.5) # The "Hidden" Biological Parameters
u0 = [0.1]                            # Initial tumor size (10% of capacity)
tspan = (0.0, 60.0)                   # 2 months trial
saveat = 0.0:4.0:60.0                 # Observations every 4 days

# 2. Run Simulation
t_obs, y_obs = generate_synthetic_data(ground_truth_dynamics!, p_true, u0, tspan, saveat, noise_level=0.05)

# 3. Visualization
p1 = scatter(t_obs, y_obs[1,:], label="Noisy Observations", 
             xlabel="Time (days)", ylabel="Tumor Volume",
             title="Synthetic Clinical Data", color=:blue, ms=6)
display(p1)

# 4. Save Data
save_path = "data/synthetic_tumor_data.jld2"
@save save_path t_obs y_obs p_true
println("Data saved to $save_path")