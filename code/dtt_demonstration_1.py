import pyAgrum as gum
import pyAgrum.lib.bn2graph as plb
import DecisionTheoreticTroubleshooting as dtt

# Exemple d'une utilisation de module DecisionTheoreticTroubleshooting qui réalise
# les algorithmes pour résoudre des problèmes de Troubleshooting
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
        'callService': 500
    }
    
    costs_obs = {
        'car.batteryFlat': 20,
        'oil.noOil': 50,
        'tank.Empty': 5,
        'tank.fuelLineBlocked': 60,
        'starter.starterBroken': 10,
    }
    
    # ... ainsi que des types des noeuds d'un réseau bayésien
    nodes_associations = {
        'car.batteryFlat': {'repairable', 'observable'},
        'oil.noOil': {'repairable', 'observable'},
        'tank.Empty': {'repairable'},
        'tank.fuelLineBlocked': {'repairable', 'observable'},
        'starter.starterBroken': {'repairable', 'observable'},
        'car.carWontStart': {'problem-defining'},
        'callService': {'service'}
    }
    # On résoudre le problème de Troubleshooting donné
    tsp = dtt.TroubleShootingProblem(bn_car, [costs_rep, costs_obs], nodes_associations)
    print(tsp.solve_static())
    #plb.pdfize(bn_car, "test")