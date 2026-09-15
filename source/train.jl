module Train

using ArcadeLearningEnvironment
using Flux
using BSON: @save
using VideoIO
using ColorTypes
using FixedPointNumbers
using Plots

include("environment.jl")
include("networks.jl")
include("agent.jl")

const SCREEN_WIDTH = 160
const SCREEN_HEIGHT = 210

function save_video(frames_rgb::Vector{Vector{UInt8}}, filename::String)
    println("Encodage de la vidéo : $filename ...")
    video_frames = [
        permutedims(
            reshape(
                reinterpret(RGB{N0f8}, frame), 
                SCREEN_WIDTH, SCREEN_HEIGHT
            ), 
            (2, 1)
        ) for frame in frames_rgb
    ]
    VideoIO.save(filename, video_frames, framerate=60)
    println("Vidéo sauvegardée.")
end

function train(; rom_path::String="pong.bin", n_frames::Int=4, n_episodes::Int=500, batch_size::Int=32)
    env = Environment.init_env(rom_path, n_frames=n_frames)
    
    actor = Networks.build_actor(n_frames)
    critic = Networks.build_critic(n_frames)
    target_critic = deepcopy(critic)
    
    opt_actor = Flux.setup(Flux.Adam(3e-4), actor)
    opt_critic = Flux.setup(Flux.Adam(3e-4), critic)
    
    fyl_loss = Agent.get_fyl_loss(0.1, 10)
    
    buffer = []
    max_buffer_size = 50_000
    
    gamma = 0.99f0
    tau = 0.1f0
    target_update_freq = 10000
    update_steps = 0
    
    episodes_to_record = [1, 50, 100, 200, 300, 500]
    episodes_to_plot = [50, 100, 300, 500]
    
    log_critic_losses_history = Float64[]
    grad_norms_history = Float64[]
    rewards_history = Float64[]
    entropies_history = Float64[] # NOUVEAU

    for episode in 1:n_episodes
        state = Environment.reset!(env)
        done = false
        episode_reward = 0.0f0
        
        episode_critic_loss = 0.0
        episode_grad_norm = 0.0
        episode_entropy = 0.0 # NOUVEAU
        updates_count = 0
        
        recording = episode in episodes_to_record
        video_frames = Vector{UInt8}[]

        while !done
            if recording
                push!(video_frames, getScreenRGB(env.ale))
            end

            action_idx = Agent.select_action(actor, state, 0.3f0)
            next_state, reward, done = Environment.step!(env, action_idx)
            episode_reward += reward
            
            if length(buffer) >= max_buffer_size
                popfirst!(buffer)
            end
            push!(buffer, (state, action_idx, reward, next_state, done))
            
            state = next_state
            
            if length(buffer) >= batch_size
                batch = rand(buffer, batch_size)
                
                states_batch = hcat([b[1] for b in batch]...)
                actions_batch = [b[2] for b in batch]
                rewards_batch = Float32.(hcat([b[3] for b in batch]...))
                next_states_batch = hcat([b[4] for b in batch]...)
                dones_batch = Float32.(hcat([b[5] for b in batch]...))
                
                # NOUVEAU : Récupération du batch_entropy
                val_actor, val_critic, grad_norm, batch_entropy = Agent.update!(actor, critic, target_critic, opt_actor, opt_critic, fyl_loss, 
                                                                 states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch, gamma, tau)
                
                episode_critic_loss += val_critic
                episode_grad_norm += grad_norm
                episode_entropy += batch_entropy
                updates_count += 1
                
                update_steps += 1
                if update_steps % target_update_freq == 0
                    Flux.loadmodel!(target_critic, critic)
                end
            end
        end
        
        avg_loss = updates_count > 0 ? (episode_critic_loss / updates_count) : 0.0
        avg_grad_norm = updates_count > 0 ? (episode_grad_norm / updates_count) : 0.0
        avg_entropy = updates_count > 0 ? (episode_entropy / updates_count) : 0.0 # NOUVEAU
        
        log_loss = log10(avg_loss + 1e-8)
        
        push!(log_critic_losses_history, log_loss)
        push!(grad_norms_history, avg_grad_norm)
        push!(rewards_history, episode_reward)
        push!(entropies_history, avg_entropy) # NOUVEAU
        
        println("Épisode $episode / $n_episodes | Score : $(round(episode_reward, digits=1)) | Loss: $(round(log_loss, digits=4)) | Grad: $(round(avg_grad_norm, digits=4)) | Entropie: $(round(avg_entropy, digits=4))")
        
        if episode % 100 == 0
            save_path = joinpath(@__DIR__, "..", "pong_weights_ep_$(episode).bson")
            @save save_path actor critic
            println("=> Sauvegarde d'étape réussie : $save_path")
        end
        
        if recording
            save_video(video_frames, "pong_agent_ep_$(episode).mp4")
        end

        if episode in episodes_to_plot
            println("=> Génération des graphiques pour l'épisode $episode...")
            x_axis = 1:episode
            
            p1 = plot(x_axis, log_critic_losses_history, title="Log Critic Loss (MSE)", xlabel="", ylabel="Log10(Loss)", legend=false, color=:red)
            p2 = plot(x_axis, grad_norms_history, title="Norme du Gradient L2", xlabel="", ylabel="Norme", legend=false, color=:purple)
            p3 = plot(x_axis, entropies_history, title="Entropie de la Politique", xlabel="", ylabel="Entropie", legend=false, color=:green)
            p4 = plot(x_axis, rewards_history, title="Score par Épisode", xlabel="Épisode", ylabel="Score", legend=false, color=:blue)
            
            # Mise en page sur 4 lignes pour que ce soit lisible
            final_plot = plot(p1, p2, p3, p4, layout=(4, 1), size=(800, 1200))
            
            plot_path = joinpath(@__DIR__, "..", "metrics_ep_$(episode).png")
            savefig(final_plot, plot_path)
            println("=> Graphiques sauvegardés : $plot_path")
        end

        GC.gc()
    end
    
    println("Entraînement terminé. Sauvegarde des poids finaux...")
    @save joinpath(@__DIR__, "..", "pong_srl_weights_final.bson") actor critic
end

end