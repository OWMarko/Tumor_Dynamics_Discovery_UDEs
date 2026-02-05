using Pkg; Pkg.activate(".")
using DifferentialEquations, SciMLSensitivity, Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays, Plots, Random, Statistics, ForwardDiff

# Importation des outils communs
include("../src/tools.jl")
include("../src/models.jl")
include("../src/physics.jl")

println("--- EXP 01: Reconstruction (Dense Data) ---")

# 1. SCÉNARIO (Doses régulières)
function drug_conc_dense(t)
    dose = 0.0
    for t_d in [10.0, 20.0, 30.0]
        (t >= t_d) && (dose += 1.0 * exp(-0.25 * (t - t_d)))
    end
    return dose
end

# 2. GÉNÉRATION DONNÉES (121 points)
u0 = [0.1]
t_dense = range(0.0, 60.0, length=121)
y_true = Array(solve(ODEProblem((u,p,t)->ground_truth_dynamics(u,p,t,drug_conc_dense), u0, (0.0, 60.0)), Vern7(), saveat=t_dense))
y_noisy = y_true .* (1.0 .+ 0.05 .* randn(Random.MersenneTwister(42), size(y_true)))

# 3. MODÈLE
rng = Random.MersenneTwister(123)
nn, p_init, st = get_optimized_ude_model(rng)

function ude_dyn(du, u, p, t)
    N = max(u[1], 1e-6)
    phys = ude_known_physics(N, 0.50) # Physique fausse
    nn_out = first(nn([N, drug_conc_dense(t)], p, st))[1]
    correction = 3.0 * ((nn_out + 1.0) / 2.0) * N # Mise à l'échelle
    du[1] = phys - smooth_abs(correction)
end

# 4. ENTRAÎNEMENT (ADAM + BFGS)
prob = ODEProblem(ude_dyn, u0, (0.0, 60.0), p_init)
function loss(p, _)
    sol = solve(prob, Vern7(), saveat=t_dense, p=p, abstol=1e-3, reltol=1e-3)
    size(sol, 2) != length(t_dense) ? 1e5 : mean(abs2, Array(sol) .- y_noisy)
end

optf = Optimization.OptimizationFunction(loss, Optimization.AutoForwardDiff())
prob_opt = Optimization.OptimizationProblem(optf, p_init)

println("-> Phase 1: ADAM...")
res1 = Optimization.solve(prob_opt, OptimizationOptimisers.Adam(0.05), maxiters=500)
println("-> Phase 2: BFGS...")
res2 = Optimization.solve(Optimization.OptimizationProblem(optf, res1.u), OptimizationOptimJL.LBFGS(), maxiters=200)

# 5. PLOT
y_pred = Array(solve(ODEProblem(ude_dyn, u0, (0.0, 60.0), res2.u), Vern7(), saveat=0.1))
p = plot(title="Fig 3: Deep Training Results")
scatter!(p, t_dense, y_noisy[1,:], label="Data (121 pts)", c=:blue, alpha=0.5)
plot!(p, 0.0:0.1:60.0, y_pred[1,:], label="UDE Fit", c=:red, lw=3)
savefig("docs/fig3_reconstruction.png")
println("Sauvegardé: docs/fig3_reconstruction.png")