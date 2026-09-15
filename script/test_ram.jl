import Pkg
Pkg.activate(@__DIR__) 

using ArcadeLearningEnvironment

function ultimate_audit_pong()
    rom_path = joinpath(@__DIR__, "pong.bin")
    ale = ALE_new()
    loadROM(ale, rom_path)
    
    # Séquence stricte pour arriver en plein milieu du jeu
    function play_to_active_state()
        reset_game(ale)
        # 1. On attend sur l'écran titre (1 sec)
        for _ in 1:60 act(ale, Int32(0)) end
        # 2. Impulsion sur FIRE (action 1) pour démarrer
        for _ in 1:5 act(ale, Int32(1)) end
        # 3. On attend très longtemps que la balle soit servie (4 sec)
        for _ in 1:250 act(ale, Int32(0)) end
    end
    
    println(">>> Exécution de la branche HAUT...")
    play_to_active_state()
    ram_base = getRAM(ale) # Capture avant mouvement
    
    # On force vers le haut pendant 15 frames
    for _ in 1:15 act(ale, Int32(2)) end
    ram_up = getRAM(ale)
    
    println(">>> Exécution de la branche BAS...")
    play_to_active_state()
    
    # On force vers le bas pendant 15 frames
    for _ in 1:15 act(ale, Int32(5)) end
    ram_down = getRAM(ale)
    
    println("\n>>> Analyse différentielle :")
    found = false
    for i in 1:128
        # On cherche l'octet qui a divergé
        if ram_up[i] != ram_down[i]
            index_doc = i - 1 
            println("=> Différence à l'index julia $i (doc $index_doc) : BASE=$(ram_base[i]), UP=$(ram_up[i]), DOWN=$(ram_down[i])")
            found = true
        end
    end
    
    if !found
        println("Toujours rien... Le jeu met vraiment beaucoup de temps à s'activer.")
    end
end

ultimate_audit_pong()