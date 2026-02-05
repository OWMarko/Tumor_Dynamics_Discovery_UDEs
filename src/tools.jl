using Plots

# In the world of differential equations, sharp corners are the enemy of stability
# The standard absolute value function has a kink at zero that confuses the solver and breaks Automatic Differentiation
# So we use a smooth approximation. Think of it as sanding down the sharp edge 
# with a tiny epsilon (1e-8) to keep the gradients flowing smoothly

function smooth_abs(x)
    return sqrt(x^2 + 1e-8)
end

# Dress up our results.
default(
    fontfamily="Computer Modern", # Style LaTeX si disponible, sinon Arial
    linewidth=2, 
    framestyle=:box, 
    grid=:alpha,
    margin=5Plots.mm,
    size=(800, 500)
)
