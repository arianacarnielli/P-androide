from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class Introduction(QWidget):
    def __init__(self, parent = None):
        QWidget.__init__(self, parent)
        
        layout = QVBoxLayout(self)

        self.title = QLabel("Bienvenu !", self) 
        
        self.title.setAlignment(Qt.AlignCenter)

        self.title.setFont(QFont('Arial', 20))
        self.startButton = QPushButton("Repair")

        layout.addWidget(self.title)
        layout.addWidget(self.startButton)
        
        