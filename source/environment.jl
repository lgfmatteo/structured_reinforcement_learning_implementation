module Environment

using ArcadeLearningEnvironment

mutable struct PongEnv
    ale::ALEPtr
    n_frames::Int
    ram_history::Matrix{Float32}
    legal_actions::Vector{Int32}
    prev_ball_x::Float32
    prev_ball_dx::Float32 #pour suivre la vélocité de la balle
end

function init_env(rom_path::String="pong.bin"; n_frames::Int=4)
    ale = ALE_new()
    loadROM(ale, rom_path)
    
    actions_utiles = Int32[0, 3, 4]
    ram_history = zeros(Float32, 128, n_frames)
    
    env = PongEnv(ale, n_frames, ram_history, actions_utiles, 0.0f0, 0.0f0)
    reset!(env)
    return env
end

function get_state(env::PongEnv)
    return vec(env.ram_history)
end

function step!(env::PongEnv, action_idx::Int)
    safe_idx = clamp(action_idx, 1, length(env.legal_actions))
    ale_action = env.legal_actions[safe_idx]
    
    reward = Float32(act(env.ale, ale_action))
    
    ram = getRAM(env.ale)
    current_ball_x = Float32(ram[50])
    current_ball_dx = current_ball_x - env.prev_ball_x
    
    #on teste si on a renvoyé une balle, pour adapter les récompenses
    if env.prev_ball_x > 130.0f0 && env.prev_ball_dx > 0.0f0 && current_ball_dx < 0.0f0 && current_ball_dx > -15.0f0
        reward += 0.2f0
    end
    
    env.prev_ball_x = current_ball_x
    env.prev_ball_dx = current_ball_dx
    
    done = game_over(env.ale)
    new_ram = Float32.(ram) ./ 255.0f0
    
    env.ram_history[:, 1:(end-1)] .= env.ram_history[:, 2:end]
    env.ram_history[:, end] .= new_ram

    return get_state(env), reward, done
end

function reset!(env::PongEnv)
    reset_game(env.ale)
    ram = getRAM(env.ale)
    new_ram = Float32.(ram) ./ 255.0f0
    
    for i in 1:env.n_frames
        env.ram_history[:, i] .= new_ram
    end
    
    env.prev_ball_x = Float32(ram[50])
    env.prev_ball_dx = 0.0f0 # Réinitialisation de la vélocité
    return get_state(env)
end

end