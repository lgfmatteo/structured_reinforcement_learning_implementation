"""Rôle : Définir les passages forward des réseaux de neurones.
L'acteur, avec comme entrée : l'état (matrice de dimension 128 x nb_frames x batch_size), et comme sortie : une matrice 3 x batch_size. 
Ce sont les scores bruts pour les 3 actions, sans fonction d'activation finale type softmax (les scores peuvent être négatifs ou positifs).
Le critique avec comme entrée : l'état (matrice 128 x nb_frames x batch_size), et comme sortie : Q_values (matrice 3 x batch_size). 
Ce réseau évalue la qualité de chaque action possible dans l'état donné."""

module Networks

using Flux

function build_actor(n_frames::Int)
    input_size = 128 * n_frames
    return Chain(
        Dense(input_size, 512, relu),
        Dense(512, 512, relu),
        Dense(512, 3)
    )
end

function build_critic(n_frames::Int)
    input_size = 128 * n_frames
    return Chain(
        Dense(input_size, 512, relu),
        Dense(512, 512, relu),
        Dense(512, 3)
    )
end

end