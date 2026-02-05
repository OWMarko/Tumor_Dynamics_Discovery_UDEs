using Plots

function smooth_abs(x)
    return sqrt(x^2 + 1e-8)
end

# Configuration globale des graphiques pour le rapport
default(
    fontfamily="Computer Modern", # Style LaTeX si disponible, sinon Arial
    linewidth=2, 
    framestyle=:box, 
    grid=:alpha,
    margin=5Plots.mm,
    size=(800, 500)
)