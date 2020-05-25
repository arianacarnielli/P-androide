from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class Troubleshoot(QWidget):
    def __init__(self, parent = None):
        QWidget.__init__(self, parent)
        
        self.layoutPrincipal = QVBoxLayout(self)
        self.layoutSecondaire = QHBoxLayout()
        self.layoutTertiaire = QHBoxLayout()
        
        self.recommendation = QLabel() 
        self.recommendation.setFont(QFont('Arial', 11))
        self.recommendation.setWordWrap(True)
        self.recommendation.setAlignment(Qt.AlignCenter)
        self.layoutPrincipal.addWidget(self.recommendation)
        
        self.layoutPrincipal.addLayout(self.layoutSecondaire)
        self.layoutPrincipal.addLayout(self.layoutTertiaire)
        
        self.listObs = QListWidget(self)
        self.listObs.setViewMode(0)
        
        self.listAct = QListWidget(self)
        self.listAct.setViewMode(0)
        
        self.obsButton = QPushButton("Observer")        
        self.obsButton.setEnabled(False)
        self.listObs.itemClicked.connect(self.obsButtonActivate)
        
        self.actButton = QPushButton("Reparer")
        self.actButton.setEnabled(False)
        self.listAct.itemClicked.connect(self.actButtonActivate)
        
        self.layoutSecondaire.addWidget(self.listObs)
        self.layoutSecondaire.addWidget(self.listAct)

        self.layoutTertiaire.addWidget(self.obsButton)
        self.layoutTertiaire.addWidget(self.actButton)

        self.eliButton = QPushButton("Repondre à des questions")
        self.layoutPrincipal.addWidget(self.eliButton)
      
    def observationsPossibles(self, eco):
        self.listObs.clear()
        for node, value in eco:
            self.listObs.addItem(QListWidgetItem("{} {:.3f}".format(node, value)))
    
    def actionsPossibles(self, ecr):
        self.listAct.clear()
        for node, value in ecr:
            self.listAct.addItem(QListWidgetItem("{} {:.3f}".format(node, value)))
            
    def obsButtonActivate(self):
        self.obsButton.setEnabled(True)

    def actButtonActivate(self):
        self.actButton.setEnabled(True)    
        
    