using Lux, Random, ComponentArrays

# L'Architecture "Anti-Stiffness" (Swish + Tanh final)
function get_optimized_ude_model(rng)
    nn = Lux.Chain(
        Lux.Dense(2, 16, swish), 
        Lux.Dense(16, 16, swish), 
        Lux.Dense(16, 1, tanh)
    )
    p, st = Lux.setup(rng, nn)
    p = ComponentArray(p) .|> Float64
    p .*= 0.1 # Initialisation
    return nn, p, st
end