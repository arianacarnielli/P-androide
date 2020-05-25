import sys
import re
import numpy as np
import pyAgrum as gum

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import DecisionTheoreticTroubleshooting as dtt

from Introduction import *
from Static import *
from Troubleshoot import *
from Observation import *
from Action import *
from Elicitation import *
from Fin import *


class MainWindow(QMainWindow):

    def __init__(self, parent = None):
        QMainWindow.__init__(self, parent)
        
###################################################
# Bayesian Network                                #
###################################################  

        # Le problème est modélisé par un réseau bayésien de PyAgrum
        bnCar = gum.loadBN("simpleCar2.bif")
            
        # On initialise les coûts des réparations et observations
        costsRep = {
            "car.batteryFlat": [100, 300],
            "oil.noOil": [50, 100],
            "tank.Empty": [40, 60],
            "tank.fuelLineBlocked": 150,
            "starter.starterBroken": [20, 60],
            "callService": 500
        }
        
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
        
        # On initialise les types des noeuds du réseau
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
        
        #On peut choisir quel algorithme utiliser entre les 4 algorithmes codés
        self.algos_possibles = [
            "simple", 
            "simple avec observations locales", 
            "myope (avec observations globales)", 
            "myope avec elicitation"
        ]
           
        
###################################################
# Propriétés de la MainWindow                     #
###################################################
        
        self.setWindowTitle("Troubleshooter")
        self.setFixedSize(600, 500) 
        
###################################################
# Differents widgets                              #
###################################################
        
        self.introduction = Introduction(self.algos_possibles)
        self.introduction.startButton.clicked.connect(self.startAlgorithme)
        
        self.static = Static()
        self.static.finButton.clicked.connect(self.finish)
        
        self.trouble = Troubleshoot() 
        self.trouble.obsButton.clicked.connect(self.callObs)
        self.trouble.actButton.clicked.connect(self.callAct)
        self.trouble.eliButton.clicked.connect(self.callEli)

        self.obs = Observation()
        self.obs.cb.activated.connect(self.makeObs)
        
        self.act = Action()
        self.act.yesButton.clicked.connect(self.makeAct)
        self.act.noButton.clicked.connect(self.makeAct)
        
        self.eli =  Elicitation()
        self.eli.yesButton.clicked.connect(self.makeEli)
        self.eli.noButton.clicked.connect(self.makeEli)
        
        self.fin = Fin()
        self.fin.finButton.clicked.connect(self.finish)
        
###################################################
# Widget principal                                #
###################################################
      
        self.stack = QStackedWidget()
        self.stack.addWidget(self.introduction)
        self.stack.addWidget(self.static)
        self.stack.addWidget(self.trouble)
        self.stack.addWidget(self.obs)
        self.stack.addWidget(self.act)
        self.stack.addWidget(self.eli)
        self.stack.addWidget(self.fin)
        
        self.setCentralWidget(self.stack)

###################################################
# Troubleshooter                                  #
################################################### 
        
        # On crée l'objet pour résoudre le problème
        self.tsp = dtt.TroubleShootingProblem(bnCar, [costsRep, costsObs], nodesAssociations)

        self.elicitationNode = ""
        self.recommendation, self.typeNodeRec, self.ecr, self.eco = self.tsp.ECR_ECO_wrapper()
        self.currentNode =  ""
        self.currentObs = ""
        self.currentAct = ""
        self.currentPossibilities = []    
        
    def startAlgorithme(self):
        self.algo = self.introduction.listAlgo.currentItem().text() 
        if self.algo == self.algos_possibles[0] or \
        self.algo == self.algos_possibles[1]:
            self.startStatic()        
        else:
            self.startTroubleshoot()
            
    def startStatic(self):
        if self.algo == self.algos_possibles[0]:
            seq, ecr = self.tsp.simple_solver()
        elif self.algo == self.algos_possibles[1]:
            seq, ecr = self.tsp.simple_solver_obs()

        text = "La séquence de réparation recommendée est la suivante, avec un cout esperé de {:.3f}.".format(ecr)
        self.static.title.setText(text)  
        self.static.showSequence(seq)
        self.stack.setCurrentWidget(self.static)
        
    def startTroubleshoot(self):
        self.trouble.observationsPossibles(self.eco)  
        self.trouble.actionsPossibles(self.ecr) 
        
        if self.typeNodeRec == "obs":
            text = "On vous recommende d'observez le composant {} avec ECO : {:.3f}".format(self.recommendation, self.eco[0][1])
        else:
            text = "On vous recommende de faire l'observation-réparation suivante : {} avec ECR : {:.3f}".format(self.recommendation, self.ecr[0][1])
        self.trouble.recommendation.setText(text)
        
        if self.algo == self.algos_possibles[2]:
            self.trouble.eliButton.setEnabled(False)
        
        self.stack.setCurrentWidget(self.trouble)
               
    def callObs(self):
        self.currentNode = re.findall('(\S+) \d+.\d+', self.trouble.listObs.currentItem().text())[0]
        self.currentPossibilities = self.tsp.bayesian_network.variable(self.currentNode).labels()        
        self.obs.resultatsPossibles(self.currentPossibilities)
        self.stack.setCurrentWidget(self.obs)        
        
    def callAct(self):     
        self.currentNode = re.findall('(\S+) \d+.\d+', self.trouble.listAct.currentItem().text())[0]
        if self.currentNode == self.tsp.service_node:
                self.act.noButton.setEnabled(False)
        
        self.stack.setCurrentWidget(self.act)
        
    def callEli(self):
        self.elicitationNode, val = self.tsp.best_EVOI()
        if not np.allclose(0, val) and val > 0:
            text = "Est-ce que le prix de réparer " + self.elicitationNode + " est plus petit que " + str(self.tsp.costs_rep[self.elicitationNode]) + " ?" 
            self.eli.title.setText(text)
            self.stack.setCurrentWidget(self.eli)       
        else:
            error = QMessageBox(((QMessageBox.Warning)), "Alerte", "Pas de questions à poser")
            error.exec()
            
    def makeObs(self, text):
        self.currentObs = self.obs.cb.currentText()
        self.tsp.add_evidence(self.currentNode, self.currentObs)
        self.recommendation, self.typeNodeRec, self.ecr, self.eco = self.tsp.ECR_ECO_wrapper()
        self.trouble.actButton.setEnabled(False)
        self.trouble.obsButton.setEnabled(False)
        
        self.startTroubleshoot()
        
    def makeAct(self):
        if self.sender().text() == "No":
            obsoletes = self.tsp.observation_obsolete(self.currentNode) 
            if self.currentNode != self.tsp.service_node:
                self.tsp.add_evidence(self.currentNode, "no")
            else:
                self.tsp.add_evidence(self.currentNode, "yes")  
            for obs in obsoletes:
                self.tsp.evidences.pop(obs)
            self.tsp.reset_bay_lp(self.tsp.evidences)
            self.recommendation, self.typeNodeRec, self.ecr, self.eco = self.tsp.ECR_ECO_wrapper()
            self.trouble.actButton.setEnabled(False)
            self.trouble.obsButton.setEnabled(False)
            
            self.startTroubleshoot()
        else:
            self.stack.setCurrentWidget(self.fin)
        
    def makeEli(self):
        if self.sender().text() == "Yes":
            islower = True
        else:
            islower = False
        self.tsp.elicitation(self.elicitationNode, islower)
        self.recommendation, self.typeNodeRec, self.ecr, self.eco = self.tsp.ECR_ECO_wrapper()
        
        self.startTroubleshoot()
           
    def finish(self):
        QApplication.exit()
        
    def quit(self):
        box = QMessageBox()
        b = box.question(self, 'Sortir ?', "Vous voulez sortir du logiciel ?", QMessageBox.Yes | QMessageBox.No)
        box.setIcon(QMessageBox.Question)
        if b == QMessageBox.Yes:
            QApplication.exit()

    def closeEvent(self, event):
        event.ignore()
        self.quit()
        

if __name__ == "__main__":
    def run_app():
        if not QApplication.instance():        
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()
        mainWin = MainWindow()
        mainWin.show()
        app.exec_()
    run_app()
