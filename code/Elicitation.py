from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class Elicitation(QWidget):
    def __init__(self, parent = None):
        QWidget.__init__(self, parent)
        
        layout = QVBoxLayout(self)

        self.title = QLabel(self) 
        self.title.setFont(QFont('Arial', 20))        
        self.title.setWordWrap(True)
        self.title.setAlignment(Qt.AlignCenter)
        
        self.yesButton = QPushButton("Yes")
        self.noButton = QPushButton("No")
        
        layout.addWidget(self.title)
        layout.addWidget(self.yesButton)
        layout.addWidget(self.noButton)
