# -*- coding: utf-8 -*-
import pyAgrum as gum
import pyAgrum.lib.bn2graph as plb
import DecisionTheoreticTroubleshooting as dtt
import numpy as np

# Exemple d'une utilisation de module DecisionTheoreticTroubleshooting qui réalise
# les algorithmes pour résoudre des problèmes de Troubleshooting
if __name__ == '__main__':

    # On suppose qu'un problème a été déjà modélisé par un réseau bayésien
    bn_car = gum.loadBN('simpleCar2.bif')
# =============================================================================
#         
#     # On initialise des coûts des réparations et des observations ...
#     costs_rep = {
#         "car.batteryFlat": [100, 300],
#         "oil.noOil": [50, 100],
#         "tank.Empty": [40, 60],
#         "tank.fuelLineBlocked": 150,
#         "starter.starterBroken": [20, 60],
#         "callService": 500
#     }
#     
#     costs_obs = {
#         "car.batteryFlat": 20,
#         "oil.noOil": 50,
#         "tank.Empty": 5,
#         "tank.fuelLineBlocked": 60,
#         "starter.starterBroken": 10,
#         "car.lightsOk": 2,
#         "car.noOilLightOn": 1,
#         "oil.dipstickLevelOk": 7
#     }
#     
#     # ... ainsi que des types des noeuds d'un réseau bayésien
#     nodes_associations = {
#         "car.batteryFlat": {"repairable", "observable"},
#         "oil.noOil": {"repairable", "observable"},
#         "tank.Empty": {"repairable"},
#         "tank.fuelLineBlocked": {"repairable", "observable"},
#         "starter.starterBroken": {"repairable", "observable"},
#         "car.lightsOk": {"unrepairable", "observable"},
#         "car.noOilLightOn": {"unrepairable", "observable"},
#         "oil.dipstickLevelOk": {"unrepairable", "observable"},
#         "car.carWontStart": {"problem-defining"},
#         "callService": {"service"}
#     }
# =============================================================================

    # On initialise les coûts des réparations et observations
    costs_rep = {
        "car.batteryFlat": [100, 300],
        "oil.noOil": [50, 100],
        "tank.Empty": [40, 60],
        "tank.fuelLineBlocked": 150,
        "starter.starterBroken": [20, 60],
        "callService": 500
    }

    costs_obs = {
        "car.batteryFlat": 20,
        "oil.noOil": 50,
        "tank.Empty": 5,
        "tank.fuelLineBlocked": 60,
        "starter.starterBroken": 10,
        "car.lightsOk": 2,
        "car.noOilLightOn": 1,
        "oil.dipstickLevelOk": 7
    }

     #On initialise les types des noeuds du réseau
# =============================================================================
#     nodes_associations = {
#          "car.batteryFlat": {"repairable", "observable"},
#          "oil.noOil": {"repairable", "observable"},
#          "tank.Empty": {"repairable"},
#          "tank.fuelLineBlocked": {"repairable", "observable"},
#          "starter.starterBroken": {"repairable", "observable"},
#          "car.lightsOk": {"unrepairable", "observable"},
#          "car.noOilLightOn": {"unrepairable", "observable"},
#          "oil.dipstickLevelOk": {"unrepairable", "observable"},
#          "car.carWontStart": {"problem-defining"},
#          "callService": {"service"}
#      }
# =============================================================================

    nodes_associations = {
        'car.batteryFlat': {'repairable', 'observable'},
        'tank.Empty': {'repairable'},
        "oil.noOil": {"repairable", "observable"},
        'oil.dipstickLevelOk': {'unrepairable', 'observable'},
        'car.carWontStart': {'problem-defining'},
        'callService': {'service'}
    }



    # On résoudre le problème de Troubleshooting donné
    tsp = dtt.TroubleShootingProblem(bn_car, [costs_rep, costs_obs], nodes_associations)
    #print(tsp.simple_solver_obs())
    #plb.pdfize(bn_car, "test")
    #true_prices = tsp.draw_true_prices()

    true_prices = tsp.costs_rep
    # {
    #     "car.batteryFlat": 120,
    #     "oil.noOil": 90,
    #     "tank.Empty": 55,
    #     "tank.fuelLineBlocked": 150,
    #     "starter.starterBroken": 50,
    #     "callService": 500
    # }
    epsilon = 0.025
    nb_min = 200
    nb_max = 10000
    
    np.savez("parametres.npz", true_prices = true_prices, epsilon = epsilon, nb_min = nb_min, nb_max = nb_max)
    
    # sortie_anti_ss, costs_ss, mean_ss, std_ss, cpt_repair_ss = tsp.simple_solver_tester(true_prices = true_prices, epsilon = epsilon, nb_min = nb_min, nb_max = nb_max) 
    # np.savez("simple_solver_tester.npz", sortie_anti = sortie_anti_ss, costs = costs_ss, mean = mean_ss, std = std_ss, cpt_repair = cpt_repair_ss)
    
    # sortie_anti_sso, costs_sso, mean_sso, std_sso, cpt_repair_sso = tsp.simple_solver_obs_tester(true_prices = true_prices, epsilon = epsilon, nb_min = nb_min, nb_max = nb_max)   
    # np.savez("simple_solver_obs_tester.npz", sortie_anti = sortie_anti_sso, costs = costs_sso, mean = mean_sso, std = std_sso, cpt_repair = cpt_repair_sso)

    sortie_anti_m, costs_m, mean_m, std_m, cpt_repair_m, cpt_obs_m = tsp.myopic_solver_tester(true_prices = true_prices, epsilon = epsilon, nb_min = nb_min, nb_max = nb_max)   
    np.savez("myopic_solver_tester.npz", sortie_anti = sortie_anti_m, costs = costs_m, mean = mean_m, std = std_m, cpt_repair = cpt_repair_m, cpt_obs = cpt_obs_m)

    # sortie_anti_me, costs_me, mean_me, std_me, cpt_repair_me, cpt_obs_me, cpt_questions_me = tsp.elicitation_solver_tester(true_prices = true_prices, epsilon = epsilon, nb_min = nb_min, nb_max = nb_max)   
    # np.savez("elicitation_solver_tester.npz", sortie_anti = sortie_anti_me, costs = costs_me, mean = mean_me, std = std_me, cpt_repair = cpt_repair_me, cpt_obs = cpt_obs_me, cpt_questions = cpt_questions_me)

    sortie_anti_b, costs_b, mean_b, std_b, cpt_repair_b, cpt_obs_b = tsp.brute_force_solver_tester(true_prices = true_prices, epsilon = epsilon, nb_min = nb_min, nb_max = nb_max)   
    np.savez("brute_force_solver_tester.npz", sortie_anti = sortie_anti_b, costs = costs_b, mean = mean_b, std = std_b, cpt_repair = cpt_repair_b, cpt_obs = cpt_obs_b)

