import numpy as np
import pyAgrum as gum
import DecisionTheoreticTroubleshooting as dtt


# Exemple d'une utilisation de module DecisionTheoreticTroubleshooting qui réalise
# les algorithmes pour résoudre des problèmes de Troubleshooting
if __name__ == '__main__':

    # On suppose qu'un problème a été déjà modélisé par un réseau bayésien
    bn_vehicle = gum.loadBN('VehicleModelTS.bif')
    # On initialise des coûts des réparations et des observations ...
    costs = {
        'car.batteryFlat': 500.0,
        'battery.batteryMeterOK': 200.0  # et etc.
    }
    # ... ainsi que des types des noeuds d'un réseau bayésien
    nodes_associations = {
        'car.batteryFlat': 'repairable',
        'battery.batteryMeterOK': 'observation'  # et etc.
    }
    # On résoudre le problème de Troubleshooting donné
    solutionTS = dtt.solve_ts_problem(bn_vehicle, costs, nodes_associations)
