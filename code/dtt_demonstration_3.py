import pyAgrum as gum
import pyAgrum.lib.bn2graph as plb
import DecisionTheoreticTroubleshooting as dtt
import StrategyTree as st
import sys
import resource

# Un autre exemple d'une utilisation de module DecisionTheoreticTroubleshooting
if __name__ == '__main__':
    # On suppose qu'un problème a été déjà modélisé par un réseau bayésien
    bn_car = gum.loadBN('simpleCar2.bif')

    # On initialise des coûts des réparations et des observations ...

    # Config de coûts #1
    costs_rep = {
        'car.batteryFlat': 200,
        'oil.noOil': 100,
        'tank.Empty': 80,
        'tank.fuelLineBlocked': 150,
        'starter.starterBroken': 40,
        'callService': 500,
        'oil.dipstickLevelOk': 7
    }

    costs_obs = {
        'car.batteryFlat': 20,
        'oil.noOil': 50,
        'tank.Empty': 5,
        'tank.fuelLineBlocked': 60,
        'starter.starterBroken': 10,
        "car.lightsOk": 2,
        "car.noOilLightOn": 1,
        "oil.dipstickLevelOk": 7
    }

    # Config de coûts #2
    # costs_rep = {
    #     'car.batteryFlat': 200,
    #     'oil.noOil': 75,
    #     'tank.Empty': 50,
    #     'tank.fuelLineBlocked': 150,
    #     'starter.starterBroken': 40,
    #     'callService': 500,
    #     'oil.dipstickLevelOk': 7
    # }
    #
    # costs_obs = {
    #     'car.batteryFlat': 20,
    #     'oil.noOil': 50,
    #     'tank.Empty': 5,
    #     'tank.fuelLineBlocked': 60,
    #     'starter.starterBroken': 10,
    #     "car.lightsOk": 2,
    #     "car.noOilLightOn": 1,
    #     "oil.dipstickLevelOk": 7
    # }

    # ... ainsi que des types des noeuds d'un réseau bayésien

    # Initialisation complète
    # nodes_associations = {
    #     'car.batteryFlat': {'repairable', 'observable'},
    #     'oil.noOil': {'repairable', 'observable'},
    #     'tank.Empty': {'repairable'},
    #     'tank.fuelLineBlocked': {'repairable', 'observable'},
    #     'starter.starterBroken': {'repairable', 'observable'},
    #     'car.lightsOk': {'unrepairable', 'observable'},
    #     'car.noOilLightOn': {'unrepairable', 'observable'},
    #     'oil.dipstickLevelOk': {'unrepairable', 'observable'},
    #     'car.carWontStart': {'problem-defining'},
    #     'callService': {'service'}
    # }

    # Initialisation courante qui est raccourcie pour ne pas surcharger un algo de recherche exhaustive
    nodes_associations = {
        'car.batteryFlat': {'repairable', 'observable'},
        'oil.noOil': {'repairable', 'observable'},
        'tank.Empty': {'repairable'},
        'tank.fuelLineBlocked': {'repairable', 'observable'},
        'starter.starterBroken': {'repairable', 'observable'},
        'car.lightsOk': {'unrepairable', 'observable'},
        'car.noOilLightOn': {'unrepairable', 'observable'},
        'oil.dipstickLevelOk': {'unrepairable', 'observable'},
        'car.carWontStart': {'problem-defining'},
        'callService': {'service'}
    }

    # Une variable boléenne qui indique si on considère des couples observation-réparation ou pas
    obs_rep_couples = True
    # TODO COMMENTAIRE
    obs_obsolete = True
    # Un nombre maximal de simulations
    nb_max = 1000
    # Une erreur permise
    epsilon = 0.01

    # Configs pour sauvegarder
    #
    # Config #1
    # nodes_associations = {
    #     'car.batteryFlat': {'repairable', 'observable'},
    #     'tank.Empty': {'repairable'},
    #     'oil.dipstickLevelOk': {'unrepairable', 'observable'},
    #     'car.carWontStart': {'problem-defining'},
    #     'callService': {'service'}
    # }
    # Config #2
    # nodes_associations = {
    #     'car.batteryFlat': {'repairable', 'observable'},
    #     'tank.Empty': {'repairable', 'observable'},
    #     'oil.dipstickLevelOk': {'unrepairable', 'observable'},
    #     'car.lightsOk': {'unrepairable', 'observable'},
    #     'car.carWontStart': {'problem-defining'},
    #     'callService': {'service'}
    # }

    tsp = dtt.TroubleShootingProblem(bn_car, [costs_rep, costs_obs], nodes_associations)

    # Un lancement de l'algo avec le dénombrement complet de tous les arbres possibles
    # best_tree, best_ecr = tsp.brute_force_solver(
    #     debug=(True, False), obs_rep_couples=obs_rep_couples, obs_obsolete=obs_obsolete)
    # print('\n\nLes meilleurs résultats pour ALL\n\n')
    # print('ECR : %f' % best_ecr)
    # print('Arbre : %s' % best_tree)
    # plb.pdfize(bn_car, "test")
    # best_tree.visualize("last_best_ST_full.gv")
    #
    # res_all = tsp.brute_force_solver_tester(
    #     costs_rep, epsilon, obs_rep_couples=obs_rep_couples, nb_max=nb_max, strategy_tree=best_tree
    # )
    # print('Le coût moyen pour une approche exacte (ALL): %f' % res_all[1])

    # Un lancement de l'algo qui utilise une programmation dynamique
    best_tree_dp, best_ecr_dp = tsp.brute_force_solver(debug=(True, False), mode='dp', obs_rep_couples=obs_rep_couples)

    print('\n\nLes meilleurs résultats pour DP\n\n')
    print('ECR : %f' % best_ecr_dp)
    print('Arbre : %s' % best_tree_dp)
    print('ECR ALT : %.16f' % tsp.expected_cost_of_repair(best_tree_dp))
    best_tree_dp.visualize()

    res_dp = tsp.brute_force_solver_tester(
        costs_rep, epsilon, obs_rep_couples=obs_rep_couples, nb_max=nb_max, strategy_tree=best_tree_dp
    )
    print('Le coût moyen pour une approche exacte (DP): %f' % res_dp[1])

    res_myopic = tsp.myopic_solver_tester(costs_rep, epsilon, nb_max=nb_max)
    print('Le coût moyen pour une approche myope : %f' % res_myopic[1])

    # L'un des arbres optimales pour une config #1 qui est différent de celui retourné si obs_rep_couples=False
    # n0 = st.Observation('0', tsp.costs_obs['oil.dipstickLevelOk'], 'oil.dipstickLevelOk')
    # n1 = st.Repair('1', tsp.costs_rep['callService'], 'callService') # n0 : no
    # n2 = st.Observation('2', tsp.costs_obs['car.batteryFlat'], 'car.batteryFlat') # n0 : yes
    # n3 = st.Observation('3', tsp.costs_obs['car.batteryFlat'], 'car.batteryFlat') # n1
    # n4 = st.Repair('4', tsp.costs_rep['tank.Empty'], 'tank.Empty') # n2 : no
    # n5 = st.Repair('5', tsp.costs_rep['car.batteryFlat'], 'car.batteryFlat') # n2 : yes
    # n6 = st.Repair('6', tsp.costs_rep['tank.Empty'], 'tank.Empty') # n3 : no
    # n7 = st.Repair('7', tsp.costs_rep['car.batteryFlat'], 'car.batteryFlat') # n3 : yes
    # n8 = st.Repair('8', tsp.costs_rep['callService'], 'callService') # n4
    # n9 = st.Repair('9', tsp.costs_rep['tank.Empty'], 'tank.Empty') # n5
    # n10 = st.Repair('10', tsp.costs_rep['tank.Empty'], 'tank.Empty') # n7
    # n11 = st.Repair('11', tsp.costs_rep['callService'], 'callService') # n9
    #
    # n9.set_child(n11)
    # n7.set_child(n10)
    # n5.set_child(n9)
    # n4.set_child(n8)
    # n3.set_yes_child(n7)
    # n3.set_no_child(n6)
    # n2.set_yes_child(n5)
    # n2.set_no_child(n4)
    # n1.set_child(n3)
    # n0.set_yes_child(n2)
    # n0.set_no_child(n1)
    #
    # test_opt_tree = st.StrategyTree(root=n0, nodes=[n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11])
    # print('\n\nTest opt tree')
    # print(test_opt_tree)
    # print(test_opt_tree.get_adj_dict())
    # print('ECR of this tree')
    # print('%.16f' % tsp.expected_cost_of_repair(test_opt_tree))
    # test_opt_tree.visualize('test_opt_tree.gv')

    # Un arbre construit à partir d'une solution retournée par une approche myope
    # n0 = st.Observation('0', tsp.costs_obs['oil.dipstickLevelOk'], 'oil.dipstickLevelOk')
    # n1 = st.Repair('1', tsp.costs_rep['callService'], 'callService')
    # n2 = st.Observation('2', tsp.costs_obs['car.batteryFlat'], 'car.batteryFlat')
    # n3 = st.Observation('3', tsp.costs_obs['tank.Empty'], 'tank.Empty')
    # n4 = st.Repair('4', tsp.costs_rep['car.batteryFlat'], 'car.batteryFlat')
    # n5 = st.Repair('5', tsp.costs_rep['callService'], 'callService')
    # n6 = st.Repair('6', tsp.costs_rep['tank.Empty'], 'tank.Empty')
    # n7 = st.Repair('7', tsp.costs_rep['callService'], 'callService')
    # n8 = st.Repair('8', tsp.costs_rep['callService'], 'callService')
    #
    # n6.set_child(n8)
    # n4.set_child(n7)
    # n2.set_yes_child(n4)
    # n2.set_no_child(n6)
    # n0.set_yes_child(n2)
    # n0.set_no_child(n1)
    #
    # test_myopic_tree = st.StrategyTree(root=n0, nodes=[n0, n1, n2, n4, n6, n7, n8])
    # test_myopic_tree.visualize('myopic_st.gv')
    # print(tsp.expected_cost_of_repair(test_myopic_tree))
    # res_myopic_test = tsp.brute_force_solver_tester(costs_rep, epsilon, nb_max=nb_max, strategy_tree=test_myopic_tree)
    # print(res_myopic_test[1])

    # Petit test de l'algo simple pour l'initialisation complète
    # nodes_associations = {
    #     'car.batteryFlat': {'repairable', 'observable'},
    #     'oil.noOil': {'repairable', 'observable'},
    #     'tank.Empty': {'repairable'},
    #     'tank.fuelLineBlocked': {'repairable', 'observable'},
    #     'starter.starterBroken': {'repairable', 'observable'},
    #     'car.lightsOk': {'unrepairable', 'observable'},
    #     'car.noOilLightOn': {'unrepairable', 'observable'},
    #     'oil.dipstickLevelOk': {'unrepairable', 'observable'},
    #     'car.carWontStart': {'problem-defining'},
    #     'callService': {'service'}
    # }
    # tsp = dtt.TroubleShootingProblem(bn_car, [costs_rep, costs_obs], nodes_associations)
    # res_simple = tsp.simple_solver_tester(costs_rep, epsilon, nb_max=nb_max)
    # print('Le coût moyen pour une approche la plus simple : %f' % res_simple[1])

    # Test d'une méthode myopic_solver_st
    # nodes_associations = {
    #     'car.batteryFlat': {'repairable', 'observable'},
    #     'oil.noOil': {'repairable', 'observable'},
    #     'tank.Empty': {'repairable'},
    #     'tank.fuelLineBlocked': {'repairable', 'observable'},
    #     'starter.starterBroken': {'repairable', 'observable'},
    #     'car.lightsOk': {'unrepairable', 'observable'},
    #     'car.noOilLightOn': {'unrepairable', 'observable'},
    #     'oil.dipstickLevelOk': {'unrepairable', 'observable'},
    #     'car.carWontStart': {'problem-defining'},
    #     'callService': {'service'}
    # }
    # tsp = dtt.TroubleShootingProblem(bn_car, [costs_rep, costs_obs], nodes_associations)
    # myopic_st = tsp.myopic_solver_st()
    # myopic_st.visualize('test_myopic_solver_st.gv')
    # res_myopic_st = tsp.brute_force_solver_tester(costs_rep, epsilon, nb_max=nb_max, strategy_tree=myopic_st)
    # print('Le coût moyen pour un arbre généré par l\'approche myope : %f' % res_myopic_st[1])
    # print(tsp.expected_cost_of_repair(myopic_st))
