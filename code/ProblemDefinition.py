# -*- coding: utf-8 -*-
"""
"""

# Nom du fichier .bif contenant le réseau Bayésien à utiliser
bnFilename = "simpleCar2.bif"

# Coûts de réparation, sous forme de dictionnaire. Les valeurs sont soit des
# nombres, soit des intervalles représentés comme un tuple ou liste de deux
# nombres.
costsRep = {
    "car.batteryFlat": [100, 300],
    "oil.noOil": [50, 100],
    "tank.Empty": [40, 60],
    "tank.fuelLineBlocked": 150,
    "starter.starterBroken": [20, 60],
    "callService": 500
}

# Coûts d'observation, sous forme de dictionnaire dont les valeurs sont des
# nombres.
costsObs = {
    "car.batteryFlat": 20,
    "oil.noOil": 50,
    "tank.Empty": 5,
    "tank.fuelLineBlocked": 60,
    "starter.starterBroken": 10,
    "car.lightsOk": 2,
    "car.noOilLightOn": 1,
    "oil.dipstickLevelOk": 7
}

# Type de chaque noeud, sous forme de dictionnaire dont les valeurs sont des
# ensembles décrivant les types de noeuds.
# Types possibles :
    # repairable
    # observable
    # unrepairable
    # problem-defining
    # service
nodesAssociations = {
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

# Des dictionnaires complémentaires de types, plus réduits,
# sont fournis pour que l'algorithme exacte s'exécute
# en temps raisonnable.

# dp, obs_rep_couples = True
nodesAssociationsSimple0 = {
    "car.batteryFlat": {"repairable", "observable"},
    "oil.noOil": {"repairable", "observable"},
    "tank.Empty": {"repairable"},
    "starter.starterBroken": {"repairable", "observable"},
    "car.lightsOk": {"unrepairable", "observable"},
    "oil.dipstickLevelOk": {"unrepairable", "observable"},
    "car.carWontStart": {"problem-defining"},
    "callService": {"service"}
}

# dp, obs_rep_couples = False
nodesAssociationsSimple1 = {
    "car.batteryFlat": {"repairable"},
    "oil.noOil": {"repairable", "observable"},
    "tank.Empty": {"repairable"},
    "starter.starterBroken": {"repairable"},
    "oil.dipstickLevelOk": {"unrepairable", "observable"},
    "car.carWontStart": {"problem-defining"},
    "callService": {"service"}
}

# all, obs_rep_couples = True
nodesAssociationsSimple2 = {
    "car.batteryFlat": {"repairable", "observable"},
    "oil.noOil": {"repairable", "observable"},
    "tank.Empty": {"repairable"},
    "oil.dipstickLevelOk": {"unrepairable", "observable"},
    "car.carWontStart": {"problem-defining"},
    "callService": {"service"}
}

# all, obs_rep_couples = False
nodesAssociationsSimple3 = {
    "car.batteryFlat": {"repairable", "observable"},
    "tank.Empty": {"repairable"},
    "oil.dipstickLevelOk": {"unrepairable", "observable"},
    "car.carWontStart": {"problem-defining"},
    "callService": {"service"}
}