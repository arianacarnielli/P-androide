from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class ShowECR(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        self.grid = QGridLayout(self)

        self.title = QLabel()
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setWordWrap(True)
        self.title.setFont(QFont('Arial', 20))
        self.title.setText("")

        self.continueButton = QPushButton("Continuer")
        self.continueButton.setEnabled(False)

        self.grid.addWidget(self.title, 0, 0)
        self.grid.addWidget(self.continueButton, 2, 0)

    def updateTitle(self, ecr):
        self.title.setText(
            "On a calculé la stratégie optimale et on a trouvé le coût espéré : %.3f" % ecr)
