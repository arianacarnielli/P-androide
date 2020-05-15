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
        "car.batteryFlat": [100, 300],
        "oil.noOil": [50, 100],
        "tank.Empty": [40, 120],
        "tank.fuelLineBlocked": 150,
        "starter.starterBroken": [20, 60],
        "callService": 500
    }
    
    costs_obs = {
        "car.batteryFlat": 1,
        "oil.noOil": 1,
        "tank.Empty": 1,
        "tank.fuelLineBlocked": 1,
        "starter.starterBroken": 1,
        "car.lightsOk": 1,
        "car.noOilLightOn": 1,
        "oil.dipstickLevelOk": 1
    }
    
    # ... ainsi que des types des noeuds d'un réseau bayésien
    nodes_associations = {
        "car.batteryFlat": {"repairable", "observable"},
        "oil.noOil": {"repairable", "observable"},
        "tank.Empty": {"repairable"},
        "tank.fuelLineBlocked": {"repairable", "observable"},
        "starter.starterBroken": {"repairable", "observable"},
        "car.lightsOk": {"unrepairable", "observable"},
        "car.noOilLightOn": {"unrepairable", "observable"},
        "oil.dipstickLevelOk": {"unrepairable", "observable"},
        "car.carWontStart": {"problem-defining"},
        "callService": {"service"}
    }
    # On résoudre le problème de Troubleshooting donné
    tsp = dtt.TroubleShootingProblem(bn_car, [costs_rep, costs_obs], nodes_associations)
    #print(tsp.solve_static())
    #plb.pdfize(bn_car, "test")