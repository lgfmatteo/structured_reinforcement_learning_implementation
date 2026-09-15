"""Rôle : transformer le vecteur de scores en une action valide a, tout en permettant la différentiation. Pas le plus intéressant dans le cas de Pong.
Entrée : theta (vecteur de taille 3 contenant les scores de l'acteur).
Sortie : a (vecteur one-hot de taille 3, ex: [0.0, 1.0, 0.0])."""

module COLayer

function solve_pong(theta::AbstractVector)
    best_idx = argmax(theta)
    a = zeros(Float32, length(theta))
    a[best_idx] = 1.0f0
    return a
end

# Version pour gérer les batchs (Matrices) envoyés par Flux
function solve_pong(Theta::AbstractMatrix)
    return hcat([solve_pong(Theta[:, i]) for i in 1:size(Theta, 2)]...)
end

end