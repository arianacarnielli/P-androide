# -*- coding: utf-8 -*-
import pyAgrum as gum
import pyAgrum.lib.bn2graph as plb
import DecisionTheoreticTroubleshooting as dtt
import numpy as np

from ProblemDefinition import * 

# Exemple d'utilisation du module DecisionTheoreticTroubleshooting qui exécute
# des algorithmes pour résoudre des problèmes de Troubleshooting
if __name__ == '__main__':

    # On suppose que le problème a été déjà modélisé par un réseau bayésien
    bn_car = gum.loadBN(bnFilename)

    # On crée l'objet pour la résolution
    tsp = dtt.TroubleShootingProblem(bn_car, [costsRep, costsObs], \
                                     nodesAssociations)

    # À partir d'ici on peut appeler l'algorithme qu'on veut, par exemple :
    print(tsp.simple_solver_obs())
    

    # On peut aussi créer de façon aléatoire des prix réels pour les 
    # réparations et appeler un testeur, par exemple (en sauvegardant les
    # résultats dans un fichier .npz) :

    true_prices = tsp.draw_true_prices()

    epsilon = 0.025
    nb_min = 200
    nb_max = 10000
    
    #np.savez("parametres.npz", true_prices = true_prices, epsilon = epsilon, nb_min = nb_min, nb_max = nb_max)
    #sortie_anti_m, costs_m, mean_m, std_m, cpt_repair_m, cpt_obs_m = tsp.myopic_solver_tester(true_prices = true_prices, epsilon = epsilon, nb_min = nb_min, nb_max = nb_max)   
    #np.savez("myopic_solver_tester.npz", sortie_anti = sortie_anti_m, costs = costs_m, mean = mean_m, std = std_m, cpt_repair = cpt_repair_m, cpt_obs = cpt_obs_m)


