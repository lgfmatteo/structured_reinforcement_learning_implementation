import Pkg
Pkg.activate(joinpath(@__DIR__, "..")) 

using BSON: @load
using VideoIO
using ColorTypes
using FixedPointNumbers
using ArcadeLearningEnvironment

include("../src/environment.jl")
include("../src/networks.jl")
include("../src/agent.jl")
include("../src/train.jl")

using .Environment
using .Networks
using .Agent
using .Train

const ROM_PATH = joinpath(@__DIR__, "..", "pong.bin")

function run_training(; n_frames=4, n_episodes=500, batch_size=32)
    println(">>> Lancement de l'entraînement (MLP, n_frames=$n_frames)")
    
    kwargs = (
        rom_path = ROM_PATH,
        n_frames = n_frames,
        n_episodes = n_episodes,
        batch_size = batch_size
    )
    
    Base.invokelatest(Main.Train.train; kwargs...)
end

function run_inference(model_name::String; n_frames=4, render::Bool=false)
    println(">>> Lancement de l'inférence : $model_name")
    
    model_path = joinpath(@__DIR__, "..", model_name)
    if !isfile(model_path)
        println("Erreur : Impossible de trouver le fichier de poids à $model_path")
        return
    end

    println("Chargement du réseau...")
    @load model_path actor
    
    env = Environment.init_env(ROM_PATH; n_frames=n_frames)
    state = Environment.reset!(env)
    
    done = false
    total_reward = 0.0
    frames_rgb = Vector{UInt8}[]
    
    println("Début de la partie...")
    while !done
        # Bruit désactivé pour exploitation pure
        action_idx = Agent.select_action(actor, state, 0.0f0)
        
        if render
            push!(frames_rgb, getScreenRGB(env.ale))
        end
        
        state, reward, done = Environment.step!(env, action_idx)
        total_reward += reward
    end
    
    println("Score final de l'IA : $total_reward")
    
    if render && !isempty(frames_rgb)
        println("Encodage de la vidéo...")
        video_frames = [
            permutedims(
                reshape(reinterpret(RGB{N0f8}, frame), 160, 210),
                (2, 1)
            ) for frame in frames_rgb
        ]
        
        mp4_path = "eval_$(replace(model_name, ".bson" => "")).mp4"
        VideoIO.save(mp4_path, video_frames, framerate=60)
        println("Vidéo sauvegardée : $mp4_path")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Décommentez la ligne que vous souhaitez exécuter :
    
    # 1. Lancer l'entraînement
    Base.invokelatest(run_training, n_frames=4, n_episodes=500, batch_size=32)
    
    # 2. Regarder un modèle spécifique jouer (ex: celui de l'épisode 200)
    # Base.invokelatest(run_inference, "pong_weights_ep_200.bson", n_frames=4, render=true)
end

run_training(n_frames=4, n_episodes=300, batch_size=256)
