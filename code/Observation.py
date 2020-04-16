from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class Observation(QWidget):
    def __init__(self, parent = None):
        QWidget.__init__(self, parent)
        
        layout = QVBoxLayout(self)

        self.title = QLabel("Quel était le résultat ?", self) 
        self.title.setFont(QFont('Arial', 20))
        self.title.setAlignment(Qt.AlignCenter)
        self.cb = QComboBox()

        layout.addWidget(self.title)
        layout.addWidget(self.cb)
        
      
    def resultatsPossibles(self, observables):
        self.cb.clear()
        self.cb.addItems(observables)
