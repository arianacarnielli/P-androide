import pyAgrum as gum
import DecisionTheoreticTroubleshooting as dtt


# Exemple d'une utilisation de module DecisionTheoreticTroubleshooting qui réalise
# les algorithmes pour résoudre des problèmes de Troubleshooting
if __name__ == '__main__':

    # On suppose qu'un problème a été déjà modélisé par un réseau bayésien
    bn_car = gum.loadBN('simpleCar.bif')
    # On initialise des coûts des réparations et des observations ...
    costs = {
        'car.batteryFlat': 200,
        'oil.noOil': 100,
        'tank.noGas': 80,
        'tank.fuelLineBlocked': 150,
        'starter.starterBroken': 40,
        'service': 500
    }
    # ... ainsi que des types des noeuds d'un réseau bayésien
    nodes_associations = {
        'car.batteryFlat': 'repairable',
        'oil.noOil': 'repairable',
        'tank.noGas': 'repairable',
        'tank.fuelLineBlocked': 'repairable',
        'starter.starterBroken': 'repairable',
        'car.carWontStart': 'problem-defining'
    }
    # On résoudre le problème de Troubleshooting donné
    tsp = dtt.TroubleShootingProblem(bn_car, costs, nodes_associations)
    print(tsp.solve_static())
