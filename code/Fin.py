from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class Fin(QWidget):
    def __init__(self, parent = None):
        QWidget.__init__(self, parent)
        
        layout = QVBoxLayout(self)

        self.title = QLabel("Merci d'avoir utilisé le logiciel !", self) 
        self.title.setFont(QFont('Arial', 20))
        self.title.setAlignment(Qt.AlignCenter)
        self.finButton = QPushButton("Ok")
        
        layout.addWidget(self.title)
        layout.addWidget(self.finButton)
        