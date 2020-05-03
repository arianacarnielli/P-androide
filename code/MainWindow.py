import sys
import numpy as np
import pyAgrum as gum

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import DecisionTheoreticTroubleshooting as dtt

from Introduction import *
from Troubleshoot import *
from Observation import *
from Action import *
from Elicitation import *
from Fin import *

class MainWindow(QMainWindow):

    def __init__(self, parent = None):
        QMainWindow.__init__(self, parent)
        
###################################################
# Propriétés de la MainWindow                     #
###################################################
        
        self.setWindowTitle("Troubleshooter")
        self.setFixedSize(600, 500) 
        
###################################################
# Differents widgets                              #
###################################################
        
        self.introduction = Introduction()
        self.introduction.startButton.clicked.connect(self.startTroubleshoot)
        
        self.trouble = Troubleshoot() 
        self.trouble.obsButton.clicked.connect(self.callObs)
        self.trouble.actButton.clicked.connect(self.callAct)
        self.trouble.eliButton.clicked.connect(self.callEli)

        self.obs = Observation()
        self.obs.cb.textActivated.connect(self.makeObs)
        
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
        self.stack.addWidget(self.trouble)
        self.stack.addWidget(self.obs)
        self.stack.addWidget(self.act)
        self.stack.addWidget(self.eli)
        self.stack.addWidget(self.fin)
        
        self.setCentralWidget(self.stack)


###################################################
# Troubleshooter                                  #
###################################################  

        # Le problème est modélisé par un réseau bayésien
        bnCar = gum.loadBN("simpleCar2.bif")
            
        # On initialise les coûts des réparations et observations
        costsRep = {
            "car.batteryFlat": [100, 300],
            "oil.noOil": [50, 100],
            "tank.Empty": [40, 120],
            "tank.fuelLineBlocked": [50, 250],
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
        # On crée l'objet pour résoudre le problème
        self.tsp = dtt.TroubleShootingProblem(bnCar, [costsRep, costsObs], nodesAssociations)

        self.repairables = self.tsp.repairable_nodes.copy()
        self.repairables.add(self.tsp.service_node)
        self.observables = set(self.tsp.observation_nodes).intersection(set(self.tsp.unrepairable_nodes))
        
        self.elicitationNode = ""
        self.recommendation, self.typeNodeRec = self.tsp.myopic_solver()
        self.currentNode =  ""
        self.currentObs = ""
        self.currentAct = ""
        self.currentPossibilities = []
        
        
    def startTroubleshoot(self):
        self.trouble.observationsPossibles(self.observables)  
        self.trouble.actionsPossibles(self.repairables) 
        
        if self.typeNodeRec == "obs":
            text = "On vous recommende d'observez le composant " + self.recommendation 
        else:
            text = "On vous recommende de faire l'observation-réparation suivante : " + self.recommendation 
        self.trouble.recommendation.setText(text)
        self.stack.setCurrentWidget(self.trouble)
        
        
    def callObs(self):
        self.currentNode = self.trouble.listObs.currentItem().text()
        self.currentPossibilities = self.tsp.bayesian_network.variable(self.currentNode).labels()        
        self.obs.resultatsPossibles(self.currentPossibilities)
        self.stack.setCurrentWidget(self.obs)        
        
    def callAct(self):
        self.stack.setCurrentWidget(self.act)
        self.currentNode = self.trouble.listAct.currentItem().text()
        
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
        self.currentObs = text
        self.tsp.change_evidence(self.currentNode, self.currentObs)
        self.recommendation, self.typeNodeRec = self.tsp.myopic_solver()
        self.observables = self.observables - {self.currentNode}
        self.trouble.actButton.setEnabled(False)
        self.trouble.obsButton.setEnabled(False)
        
        self.startTroubleshoot()
        
    def makeAct(self):
        if self.sender().text() == "No":
            obsoletes = self.tsp.observation_obsolete(self.currentNode) 
            if self.currentNode != "callService":
                self.tsp.change_evidence(self.currentNode, "no")
            else:
                self.tsp.change_evidence(self.currentNode, "yes")  
            for obs in obsoletes:
                self.tsp.evidences.pop(obs)
            self.tsp.reset_bay_lp(self.tsp.evidences)
            self.observables.update(obsoletes)
            self.recommendation, self.typeNodeRec = self.tsp.myopic_solver()
            self.repairables = self.repairables - {self.currentNode}
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
        self.startTroubleshoot()
    
        
    def finish(self):
        sys.exit()
        
    def quit(self):
        box = QMessageBox()
        b = box.question(self, 'Sortir ?', "Vous voulez sortir du logiciel ?", QMessageBox.Yes | QMessageBox.No)
        box.setIcon(QMessageBox.Question)
        if b == QMessageBox.Yes:
            sys.exit()

    def closeEvent(self, event):
        event.ignore()
        self.quit()

        
if __name__=="__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    app.exec_()
