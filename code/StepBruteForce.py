from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class StepBruteForce(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        layout = QVBoxLayout(self)

        self.title = QLabel()
        self.title.setFont(QFont('Arial', 20))
        self.title.setAlignment(Qt.AlignCenter)

        self.okButton = QPushButton("Ok")

        layout.addWidget(self.title)
        layout.addWidget(self.okButton)

    def setTitle(self, title):
        self.title.setText(title)
