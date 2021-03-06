# -*- coding: utf-8 -*-
import sys
import re
import numpy as np
import pyAgrum as gum
import time as time
import StrategyTree as st
import socket as socket
import os as os
from multiprocessing import Process


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
from ConfigBruteForce import *
from ShowECR import *
from StepBruteForce import *

from ProblemDefinition import *

class MainWindow(QMainWindow):

    def __init__(self, parent = None):
        QMainWindow.__init__(self, parent)

        # Le problème est modélisé par un réseau bayésien de PyAgrum
        self.bnCarFilename = bnFilename
        bnCar = gum.loadBN(self.bnCarFilename)

        # On initialise les coûts des réparations et observations
        self.costsRep = costsRep
        self.costsObs = costsObs

        # Une initialisation raccourcie pour ne pas surcharger des algorithmes exactes
        self.nodesAssociations = nodesAssociations

        #On peut choisir quel algorithme utiliser entre les 5 algorithmes codés

        self.algos_possibles = [
            "simple",
            "simple avec observations locales",
            "myope (avec observations globales)",
            "myope avec elicitation",
            "recherche exhaustive"
        ]
        self.size = (600, 500)
        self.configSize = (300, 350)
        self.progressSize = (500, 200)

###################################################
# Propriétés de la MainWindow                     #
###################################################

        self.setWindowTitle("Troubleshooter")
        self.resize(self.size[0], self.size[1])

###################################################
# Differents widgets                              #
###################################################

        self.introduction = Introduction(self.algos_possibles)
        self.introduction.startButton.clicked.connect(self.startAlgorithme)

        self.static = Static()
        self.static.finButton.clicked.connect(self.fin)

        self.trouble = Troubleshoot()
        self.trouble.obsButton.clicked.connect(self.callObs)
        self.trouble.actButton.clicked.connect(self.callAct)
        self.trouble.eliButton.clicked.connect(self.callEli)

        self.obs = Observation()
        self.obs.cb.activated.connect(self.makeObs)

        self.act = Action()
        self.act.yesButton.clicked.connect(self.makeAct)
        self.act.noButton.clicked.connect(self.makeAct)

        self.eli = Elicitation()
        self.eli.yesButton.clicked.connect(self.makeEli)
        self.eli.noButton.clicked.connect(self.makeEli)

        self.fin = Fin()
        self.fin.finButton.clicked.connect(self.finish)

        self.config = ConfigBruteForce()
        self.config.calcButton.clicked.connect(self.calculateBF)
        self.config.progressBar.valueChanged.connect(self.pbarChanged)

        self.showECR = ShowECR()
        self.showECR.continueButton.clicked.connect(self.continueWithBF)

        self.step = StepBruteForce()
        self.step.okButton.clicked.connect(self.stepOk)

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
        self.stack.addWidget(self.config)
        self.stack.addWidget(self.showECR)
        self.stack.addWidget(self.step)

        self.setCentralWidget(self.stack)

###################################################
# Troubleshooter                                  #
###################################################

        # On crée l'objet pour résoudre le problème
        self.tsp = dtt.TroubleShootingProblem(bnCar, [self.costsRep, self.costsObs], self.nodesAssociations)

        self.repairables = self.tsp.repairable_nodes.copy()
        self.repairables.add(self.tsp.service_node)
        self.observables = set(self.tsp.observation_nodes).intersection(set(self.tsp.unrepairable_nodes))

        self.elicitationNode = ""
        self.recommendation, self.typeNodeRec, self.ecr, self.eco = self.tsp.ECR_ECO_wrapper()
        self.currentNode =  ""
        self.currentObs = ""
        self.currentAct = ""
        self.currentPossibilities = []

        self.optimalStrategyTree = None
        self.optimalStrategyTreeCopy = None
        self.optimalECR = self.costsRep[self.tsp.service_node]
        self.obsRepCouples = None
        self.obsObsolete = None
        self.modeCalc = None
        self.modeExec = ""
        self.bruteForce = False
        self.bruteForceStats = {}
        self.exchangeFileName = "optimal_strategy_tree.txt"
        self.bfProcess = None
        self.randomSocketPort = None

    def startAlgorithme(self):
        self.algo = self.introduction.listAlgo.currentItem().text()
        if self.algo == self.algos_possibles[0] or \
        self.algo == self.algos_possibles[1]:
            self.startStatic()
        elif self.algo == self.algos_possibles[4]:
            self.bruteForce = True
            self.startBruteForce()
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

    def startBruteForce(self):
        # answer = QMessageBox.question(
        #     self, "Attention !",
        #     "Les calculs avec la recherche exhaustive peuvent être trop"
        #     " lourds (à peu près 50 minutes même pour la meilleure "
        #     "configuration). Voulez-vous utiliser une version simplifiée "
        #     "du problème ?",
        #     QMessageBox.Yes | QMessageBox.No)
        # if answer == QMessageBox.Yes:
        #     self.nodesAssociations = nodesAssociationsSimple3
        #     self.tsp = dtt.TroubleShootingProblem(
        #         gum.loadBN(self.bnCarFilename), [self.costsRep, self.costsObs],
        #         self.nodesAssociations)
        self.resize(self.configSize[0], self.configSize[1])
        self.bruteForceStats["rep_num"] = 0
        self.bruteForceStats["obs_num"] = 0
        self.bruteForceStats["ecr"] = 0.0
        self.stack.setCurrentWidget(self.config)

    def callObs(self):
        if not self.bruteForce:
            self.currentNode = re.findall('(\S+) \d+.\d+', self.trouble.listObs.currentItem().text())[0]
        else:
            self.currentNode = self.optimalStrategyTreeCopy.get_root().get_name()
        self.currentPossibilities = self.tsp.bayesian_network.variable(self.currentNode).labels()
        self.obs.resultatsPossibles(self.currentPossibilities)
        self.stack.setCurrentWidget(self.obs)

    def callAct(self):
        if not self.bruteForce:
            self.currentNode = re.findall('(\S+) \d+.\d+', self.trouble.listAct.currentItem().text())[0]
        else:
            self.currentNode = self.optimalStrategyTreeCopy.get_root().get_name()
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
        if not self.bruteForce:
            self.tsp.add_evidence(self.currentNode, self.currentObs)
            self.recommendation, self.typeNodeRec, self.ecr, self.eco = self.tsp.ECR_ECO_wrapper()
            self.trouble.actButton.setEnabled(False)
            self.trouble.obsButton.setEnabled(False)

            self.startTroubleshoot()
        else:
            self.passToNextStep(self.currentObs)
            if self.optimalStrategyTreeCopy is None:
                self.optimalStrategyTreeCopy = st.StrategyTree(
                    root=st.Repair('0', self.tsp.costs_rep[self.tsp.service_node], self.tsp.service_node))
            self.showCurrentNodeBF()

    def makeAct(self):
        if self.sender().text() == "No":
            if not self.bruteForce:
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
                self.passToNextStep()
                if self.optimalStrategyTreeCopy is None:
                    self.optimalStrategyTreeCopy = st.StrategyTree(
                        root=st.Repair('0', self.tsp.costs_rep[self.tsp.service_node], self.tsp.service_node))
                self.showCurrentNodeBF()
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
        if self.bruteForce and self.modeExec == "step-by-step":
            print(self.bruteForceStats)
        QApplication.exit()
    
    def fin(self):
        self.stack.setCurrentWidget(self.fin)

    def calculateBF(self):
        self.config.calcButton.setEnabled(False)
        self.obsRepCouples = self.config.checkObsRepCouples.isChecked()
        self.obsObsolete = self.config.checkObsObsObsolete.isChecked()
        if self.config.radioCalcAll.isChecked():
            self.modeCalc = "all"
        else:
            self.modeCalc = "dp"
        if self.config.radioExecStepByStep.isChecked():
            self.modeExec = "step-by-step"
        else:
            self.modeExec = "show-tree"
        answer = QMessageBox.question(
            self, "Attention !",
            "Les calculs avec la recherche exhaustive peuvent être trop"
            " lourds (à peu près 50 minutes même pour la meilleure "
            "configuration). Voulez-vous utiliser une version simplifiée "
            "du problème ?",
            QMessageBox.Yes | QMessageBox.No)
        if answer == QMessageBox.Yes:
            if self.modeCalc == "dp" and self.obsRepCouples:
                self.nodesAssociations = nodesAssociationsSimple0
            elif self.modeCalc == "dp" and not self.obsRepCouples:
                self.nodesAssociations = nodesAssociationsSimple1
            elif self.modeCalc == "all" and self.obsRepCouples:
                self.nodesAssociations = nodesAssociationsSimple2
            elif self.modeCalc == "all" and not self.obsRepCouples:
                self.nodesAssociations = nodesAssociationsSimple3
            self.tsp = dtt.TroubleShootingProblem(
                gum.loadBN(self.bnCarFilename), [self.costsRep, self.costsObs],
                self.nodesAssociations)
        pbarMax = self.findPbarMax()
        self.config.progressBar.setRange(0, pbarMax)
        self.randomSocketPort = int(np.random.randint(1024, 10000, 1))
        if os.name == "nt":
            self.bfProcess = Process(
                target=launch_brute_force_multi_processing_windows,
                args=(
                    self.bnCarFilename, [self.costsRep, self.costsObs], self.nodesAssociations, self.randomSocketPort,
                    self.modeCalc, self.obsRepCouples, self.obsObsolete, self.exchangeFileName
                )
            )
        else:
            self.bfProcess = Process(target=self.launchBruteForceMultiProcessing)
        self.config.calcButton.setText("Le calcul de la stratégie optimale est en cours...")
        self.bfProcess.start()
        self.managePbar()

    def pbarChanged(self, val):
        if self.config.progressBar.maximum() == val:
            self.optimalStrategyTree, self.optimalECR = st.st_from_file(self.exchangeFileName)
            self.showECR.continueButton.setEnabled(True)
            self.showECR.updateTitle(self.optimalECR)
            self.resize(self.progressSize[0], self.progressSize[1])
            self.stack.setCurrentWidget(self.showECR)

    def continueWithBF(self):
        if self.modeExec == "show-tree":
            self.stack.setCurrentWidget(self.fin)
            ost_filename = "optimal_strategy_tree.gv"
            if isinstance(self.optimalStrategyTree, st.StrategyTree):
                self.optimalStrategyTree.visualize(ost_filename)
        elif self.modeExec == "step-by-step":
            self.optimalStrategyTreeCopy = self.optimalStrategyTree.copy()
            self.resize(self.size[0], self.size[1])
            self.showCurrentNodeBF()

    def stepOk(self):
        if isinstance(self.optimalStrategyTreeCopy.get_root(), st.Observation):
            self.bruteForceStats["obs_num"] += 1
            self.bruteForceStats["ecr"] += self.tsp.costs_obs[self.optimalStrategyTreeCopy.get_root().get_name()]
            self.callObs()
        else:
            self.bruteForceStats["rep_num"] += 1
            self.bruteForceStats["ecr"] += self.tsp.costs_rep[self.optimalStrategyTreeCopy.get_root().get_name()]
            self.callAct()

    def showCurrentNodeBF(self):
        node = (
            self.optimalStrategyTreeCopy.get_root()
            if isinstance(self.optimalStrategyTreeCopy, st.StrategyTree) else None)
        if node is None:
            self.hide()
            msg = QMessageBox(
                QMessageBox.Critical, "Erreur critique",
                "Une erreur critique s'est produite ! L'application se terminera !", QMessageBox.Ok)
            msg.exec_()
            QApplication.exit()
            return
        node_name = node.get_name()
        node_type = ("réparation" if isinstance(node, st.Repair) else "observation")
        self.step.setTitle("Veuillez exécuter une %s \"%s\"" % (node_type, node_name))
        self.stack.setCurrentWidget(self.step)

    def passToNextStep(self, obsRes=None):
        self.optimalStrategyTreeCopy = self.optimalStrategyTreeCopy.get_sub_tree(
            self.optimalStrategyTreeCopy.get_node(
                self.optimalStrategyTreeCopy.get_root()
            ).get_child_by_attribute(obsRes)
        )

    def launchBruteForceMultiProcessing(self):
        sock = socket.socket()
        sock.connect(("localhost", self.randomSocketPort))
        sock.send("0".encode())
        best_tree, best_ecr = self.tsp.brute_force_solver(
            mode=self.modeCalc, obs_rep_couples=self.obsRepCouples, obs_obsolete=self.obsObsolete,
            sock=sock
        )
        filename = self.exchangeFileName
        best_tree.to_file(filename)
        fout = open(filename, "a")
        fout.write(best_tree.fout_newline + str(best_ecr) + best_tree.fout_newline)
        fout.close()

        sock.send("1".encode())
        sock.close()

    def managePbar(self):
        sock = socket.socket()
        sock.bind(("", self.randomSocketPort))
        sock.listen(1)
        conn, addr = sock.accept()
        while self.config.progressBar.value() < self.config.progressBar.maximum():
            data = conn.recv(50).decode()
            if "0" in data:
                self.config.progressBar.setValue(
                    self.config.progressBar.value() + data.count("0")
                    if self.config.progressBar.value() + data.count("0") < self.config.progressBar.maximum()
                    else self.config.progressBar.maximum() - 1
                )
                QApplication.processEvents()
            if "1" in data:
                self.config.progressBar.setValue(self.config.progressBar.maximum())
                QApplication.processEvents()
        conn.close()

    def findPbarMax(self):
        pbarMax = 0
        fnodesNum = (
                len(self.tsp.repairable_nodes.union(self.tsp.observation_nodes)) + 1
                if self.obsRepCouples else
                len(self.tsp.repairable_nodes) + len(self.tsp.observation_nodes) + 1
        )
        if self.obsRepCouples and self.modeCalc == "dp":
            for _ in self.tsp.repairable_nodes:
                pbarMax += fnodesNum - 1
            for node_name in self.tsp.observation_nodes:
                if node_name not in self.tsp.repairable_nodes:
                    pbarMax += 2 * (fnodesNum - 1)

        elif self.obsRepCouples and self.modeCalc == "all":
            pbarMax += fnodesNum * (fnodesNum - 1)

        elif not self.obsRepCouples and self.modeCalc == "dp":
            for _ in self.tsp.repairable_nodes:
                pbarMax += fnodesNum - 1
            for _ in self.tsp.observation_nodes:
                pbarMax += 2 * (fnodesNum - 1)

        elif not self.obsRepCouples and self.modeCalc == "all":
            pbarMax += fnodesNum * (fnodesNum - 1)

        return pbarMax + 2

    def quit(self):
        box = QMessageBox()
        b = box.question(self, 'Sortir ?', "Vous voulez sortir du logiciel ?", QMessageBox.Yes | QMessageBox.No)
        box.setIcon(QMessageBox.Question)
        if b == QMessageBox.Yes:
            QApplication.exit()

    def closeEvent(self, event):
        event.ignore()
        if self.bfProcess is not None:
            self.bfProcess.join()
        self.quit()


def launch_brute_force_multi_processing_windows(
        bayesian_network_filename, costs, nodes_types, port, mode_calc, obs_rep_couples, obs_obsolete,
        exchange_file_name
):
    tsp = dtt.TroubleShootingProblem(gum.loadBN(bayesian_network_filename), costs, nodes_types)
    sock = socket.socket()
    sock.connect(("localhost", port))
    sock.send("0".encode())
    best_tree, best_ecr = tsp.brute_force_solver(
        mode=mode_calc, obs_rep_couples=obs_rep_couples, obs_obsolete=obs_obsolete, sock=sock
    )
    filename = exchange_file_name
    best_tree.to_file(filename)
    fout = open(filename, "a")
    fout.write(best_tree.fout_newline + str(best_ecr) + best_tree.fout_newline)
    fout.close()

    sock.send("1".encode())
    sock.close()


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
