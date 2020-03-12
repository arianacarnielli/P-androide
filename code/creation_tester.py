# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 00:35:25 2020

@author: arian
"""
import pyAgrum.lib.bn2graph as plb
import pyAgrum as gum
import os

#bn = gum.BayesNet("tester1")

#a = bn.add(gum.LabelizedVariable("a.broken"))
#b = bn.add(gum.LabelizedVariable("b.broken"))
#c = bn.add(gum.LabelizedVariable("c.broken"))
#service = bn.add(gum.LabelizedVariable("callService"))
#tester = bn.add(gum.LabelizedVariable("tester.notWorking"))

#bn.addArc(service, tester)
#bn.addArc(a, tester)
#bn.addArc(b, tester)
#bn.addArc(c, b)

#bn.cpt(service).fillWith([1, 0])
#bn.cpt(a).fillWith([0.75, 0.25])
#bn.cpt(c).fillWith([0.5, 0.5])

#gum.saveBN(bn, "tester1.bif")

if __name__ == '__main__':

    # On suppose qu'un problème a été déjà modélisé par un réseau bayésien
    bn_tester = gum.loadBN('tester.bif')
    
    plb.pdfize(bn_tester, "tester")