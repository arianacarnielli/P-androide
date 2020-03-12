# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 00:56:40 2020

@author: arian
"""

import pyAgrum as gum
import pyAgrum.lib.bn2graph as plb
import DecisionTheoreticTroubleshooting as dtt

# Exemple d'une utilisation de module DecisionTheoreticTroubleshooting qui réalise
# les algorithmes pour résoudre des problèmes de Troubleshooting
if __name__ == '__main__':

    # On suppose qu'un problème a été déjà modélisé par un réseau bayésien
    bn_tester = gum.loadBN("tester.bif")
        
    # On initialise des coûts des réparations et des observations ...
    costs_rep = {
        "a.broken": 200,
        "b.broken": 150,
        'callService': 500
    }
    
    costs_obs = {
        "a.broken": 10,
        "b.broken": 60,
        "c.broken": 10
    }
    
    # ... ainsi que des types des noeuds d'un réseau bayésien
    nodes_associations = {
        "a.broken": {'repairable', 'observable'},
        "b.broken": {'repairable', 'observable'},
        "c.broken": {"observable", "unrepairable"},
        "tester.notWorking": {'problem-defining'},
        'callService': {'service'}
    }
    # On résoudre le problème de Troubleshooting donné
    tsp = dtt.TroubleShootingProblem(bn_tester, [costs_rep, costs_obs], nodes_associations)
    print(tsp.solve_static())
    #plb.pdfize(bn_tester, "tester")