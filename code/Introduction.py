from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class Introduction(QWidget):
    def __init__(self, listAlgo, parent = None):
        QWidget.__init__(self, parent)

        self.layout = QVBoxLayout(self)

        self.title = QLabel("Choisissez l'algorithme utilisé", self)    
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setWordWrap(True)
        self.title.setFont(QFont('Arial', 20))
        
        self.listAlgo = QListWidget(self)
        self.listAlgo.setViewMode(0)
        
        for algo in listAlgo:
            self.listAlgo.addItem(QListWidgetItem(algo))
        self.listAlgo.itemClicked.connect(self.startButtonActivate) 
        
        self.startButton = QPushButton("Repair")
        self.startButton.setEnabled(False)

        self.layout.addWidget(self.title)
        self.layout.addWidget(self.listAlgo)
        self.layout.addWidget(self.startButton)
        
    def startButtonActivate(self):
        self.startButton.setEnabled(True)
        