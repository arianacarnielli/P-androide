# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 23:56:44 2020

@author: arian
"""
import pyAgrum as gum
import pyAgrum.lib.bn2graph as plb
import DecisionTheoreticTroubleshooting as dtt

# Exemple d'une utilisation de module DecisionTheoreticTroubleshooting qui réalise
# les algorithmes pour résoudre des problèmes de Troubleshooting
if __name__ == '__main__':

    # On suppose qu'un problème a été déjà modélisé par un réseau bayésien
    bn_tester = gum.fastBN("U{yes|no}->A{yes|no}->B{yes|no}->C{yes|no};"\
                           "I{yes|no}->D{yes|no}->R{yes|no}->E{yes|no}<-B{yes|no};"\
                           "N{yes|no}<-F{yes|no}<-C{yes|no}->G{yes|no}->H{yes|no}->P{yes|no};"\
                           "J{yes|no}->K{yes|no}<-B{yes|no};"\
                           "L{yes|no}->M{yes|no}->O{yes|no}->B{yes|no};"\
                           "H{yes|no}<-S{yes|no}<-T{yes|no}")
        
    # On initialise des coûts des réparations et des observations ...
    #costs_rep = {
    #    "a.broken": 200,
    #    "b.broken": 150,
    #    'callService': 500
    #}
    
    #costs_obs = {
    #    "a.broken": 10,
    #    "b.broken": 60,
    #    "c.broken": 10
    #}
    
    # ... ainsi que des types des noeuds d'un réseau bayésien
    nodes_associations = {
        "I": {'unrepairable', 'observable'},
        "M": {'unrepairable', 'observable'},
        "U": {"observable", "unrepairable"},
        "L": {"observable", "unrepairable"},
        "O": {"observable", "repairable"},
        "F": {'unrepairable', 'observable'},
        "C": {'repairable', 'observable'},
        "E": {'repairable'},
        "P": {"problem-defining"},
        "S": {"service"}
    }
    # On résoudre le problème de Troubleshooting donné
    tsp = dtt.TroubleShootingProblem(bn_tester, [costs_rep, costs_obs], nodes_associations)
    #print(tsp.solve_static())
    #plb.pdfize(bn_tester, "reseau_3")
