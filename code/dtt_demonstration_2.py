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

    # ... ainsi que des types des noeuds d'un réseau bayésien
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
    nodes_associations = {
        # 'car.batteryFlat': {'repairable', 'observable'},
        'tank.Empty': {'repairable'},
        'oil.dipstickLevelOk': {'unrepairable', 'observable'},
        'car.carWontStart': {'problem-defining'},
        'callService': {'service'}
    }
    tsp = dtt.TroubleShootingProblem(bn_car, [costs_rep, costs_obs], nodes_associations)

    # pot = tsp.bay_lp.posterior(tsp.problem_defining_node)
    # inst = gum.Instantiation(pot)
    # inst.chgVal(tsp.problem_defining_node, "no")
    # prob = (1 - pot[inst])
    # print(prob)

    # On construit une stratégie par mains et on calcule son ECR
    #
    # n__1 = st.Observation('-1', costs_obs['car.batteryFlat'], 'car.batteryFlat')
    # n0 = st.Repair('0', costs_rep['tank.fuelLineBlocked'], 'tank.fuelLineBlocked')
    # n1 = st.Repair('1', costs_rep['car.batteryFlat'], 'car.batteryFlat')
    # n2 = st.Repair('2', costs_rep['oil.noOil'], 'oil.noOil')
    # n3 = st.Repair('3', costs_rep['tank.Empty'], 'tank.Empty')
    # n4 = st.Repair('4', costs_rep['car.batteryFlat'], 'car.batteryFlat')
    #
    # n__1.set_no_child(n0)
    # n__1.set_yes_child(n4)
    # n0.set_child(n1)
    # n1.set_child(n2)
    # n2.set_child(n3)
    #
    # st0 = st.StrategyTree(n__1, [n__1, n0, n1, n2, n3, n4])
    # ecr = tsp.expected_cost_of_repair(st0)
    # print(ecr)
    # print(tsp.expected_cost_of_repair_seq_of_actions(
    #     ['tank.fuelLineBlocked', 'car.batteryFlat', 'oil.noOil', 'tank.Empty']
    # ))
    best_tree, best_ecr = tsp.brute_force_solver(debug=(True, False))
    print('\n\nLes meilleurs résultats pour ALL\n\n')
    print('ECR : %f' % best_ecr)
    print('Arbre : %s' % best_tree)
    # for par, ch, attr in best_tree.get_edges():
    #     print('(%s, %s, %s)' % (par.get_name(), ch.get_name(), attr if attr is not None else 'None'))
    # plb.pdfize(bn_car, "test")

    best_tree_dp, best_ecr_dp = tsp.brute_force_solver(debug=(True, False), mode='dp')

    print('\n\nLes meilleurs résultats pour DP\n\n')
    print('ECR : %f' % best_ecr_dp)
    print('Arbre : %s' % best_tree_dp)
    print('ECR ALT : %f' % tsp.expected_cost_of_repair(best_tree_dp))
    # n0_temp = st.Repair('-1', 0, 'car.carbatteryFlat')
    # n1_temp = st.Repair('-2', 0, 'callService')
    # n0_temp.set_child(n1_temp)
    # tree_temp = st.StrategyTree(root=n0_temp, nodes=[n0_temp, n1_temp])
    # print(tree_temp)
    # print(tree_temp.connect(best_tree, 'no'))
