# La Physique "Fausse" (Surestimée) donnée à l'UDE
function ude_known_physics(N, p_growth=0.50)
    return p_growth * N * log(max(1.0001, 1.0 / max(N, 1e-6)))
end

# La Vraie Physique (Pour générer les données)
function ground_truth_dynamics(u, p, t, dose_func)
    N = u[1]
    growth = 0.35 * N * log(max(1.0 / max(N, 1e-6), 1.0001))
    kill = 2.5 * dose_func(t) * N
    
    # CORRECTION ICI : On met des crochets pour renvoyer un Vecteur
    return [growth - kill] 
end