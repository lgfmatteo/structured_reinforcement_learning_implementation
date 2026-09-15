module Agent

using Flux
using InferOpt
using Random

include("co_layer.jl")

function get_fyl_loss(sigma_b::Float64, m_samples::Int)
    perturbed_optimizer = PerturbedAdditive(COLayer.solve_pong; ε=sigma_b, nb_samples=m_samples)
    return FenchelYoungLoss(perturbed_optimizer)
end

function select_action(actor, state::AbstractVector, sigma_f::Float32)
    theta = actor(state)
    eta = theta .+ randn(Float32, size(theta)) .* sigma_f
    a_one_hot = COLayer.solve_pong(eta)
    return argmax(a_one_hot) 
end

function compute_grad_norm(grads)
    n = 0.0f0
    function traverse(g)
        if g isa AbstractArray
            n += sum(abs2, g)
        elseif g isa NamedTuple || g isa Tuple
            for v in g
                traverse(v)
            end
        end
    end
    traverse(grads)
    return sqrt(n)
end

function update!(actor, critic, target_critic, opt_actor, opt_critic, fyl_loss, states, actions_idx, rewards, next_states, dones, gamma::Float32, tau::Float32)
    
    val_critic, grads_critic = Flux.withgradient(critic) do c
        q_next = target_critic(next_states)
        max_q_next = maximum(q_next, dims=1)
        
        targets = rewards .+ gamma .* max_q_next .* (1.0f0 .- dones)
        
        q_current = c(states)
        action_mask = Flux.onehotbatch(actions_idx, 1:3)
        q_played = sum(q_current .* action_mask, dims=1)
        
        Flux.mse(q_played, targets)
    end
    Flux.update!(opt_critic, critic, grads_critic[1])

    grad_norm = compute_grad_norm(grads_critic[1])

    #calcul des cibles pour l'Acteur
    q_values = critic(states)
    a_hat = Flux.softmax(q_values ./ tau, dims=1)

    #mise à jour de l'Acteur
    val_actor, grads_actor = Flux.withgradient(actor) do a
        theta = a(states)
        fyl_loss(theta, a_hat)
    end
    Flux.update!(opt_actor, actor, grads_actor[1])

    theta_out = actor(states)
    probs = Flux.softmax(theta_out, dims=1)
    
    
    entropies = -sum(probs .* log.(probs .+ 1f-8), dims=1)
    
    # On fait la moyenne sur la taille du batch
    avg_batch_entropy = sum(entropies) / length(entropies)

    return val_actor, val_critic, grad_norm, avg_batch_entropy
end

end