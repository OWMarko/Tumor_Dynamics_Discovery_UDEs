using Lux, Random, ComponentArrays

# We build the Brain of our UDE here
# It takes 2 inputs tumor Size (N) and Drug Concentration (C)

function get_optimized_ude_model(rng)
    nn = Lux.Chain(

        Lux.Dense(2, 16, swish),     # ODE solvers hate sharp edges. Swish is smooth.
        Lux.Dense(16, 16, swish),
        
        # tanh bounds the output between [-1, 1]
        # This prevents the network from predicting huge values that crash the solver (anti stiff)
        Lux.Dense(16, 1, tanh)
    )

    p, st = Lux.setup(rng, nn)

    # force Float64 (Double Precision)
    # Neural nets usually run on Float32 but Differential Equations require exactness
    p = ComponentArray(p) .|> Float64

    # This makes the Neural Net "quiet" at the start allowing the Physics to lead first
    p .*= 0.1

    return nn, p, st
end
