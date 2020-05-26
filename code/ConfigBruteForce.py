from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class ConfigBruteForce(QWidget):
    def __init__(self, parent = None):
        QWidget.__init__(self, parent)

        self.grid = QGridLayout()

        self.title = QLabel()
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setWordWrap(True)
        self.title.setFont(QFont('Arial', 20))
        self.title.setText("Veuillez remplir une configuration de l'algorithme")

        self.checkObsRepCouples = QCheckBox("Considérer des couples observations-réparations")

        self.checkObsObsObsolete = QCheckBox("Considérer des observations obsolètes")

        self.radioCalcAll = QRadioButton("Dénombrement complet")

        self.radioCalcDP = QRadioButton("Programmation dynamique")

        self.radioExecStepByStep = QRadioButton("Pas-à-pas")

        self.radioExecTree = QRadioButton("Afficher un arbre")

        self.calcButton = QPushButton("Calculer")

        # self.modeCalc = QRadioButton()
        # self.modeCalc.setText("Mode de calcul")

        self.grid.addWidget(self.title, 0, 0, 1, 2)
        self.grid.addWidget(self.createFirstGroup(), 1, 0, 1, 2)
        self.grid.addWidget(self.createSecondGroup(), 2, 0)
        self.grid.addWidget(self.createThirdGroup(), 2, 1)
        self.grid.addWidget(self.calcButton, 3, 0, 1, 2)
        self.setLayout(self.grid)

    def createFirstGroup(self):
        groupBox = QGroupBox()

        vbox = QVBoxLayout()
        vbox.addWidget(self.checkObsRepCouples)
        vbox.addWidget(self.checkObsObsObsolete)
        groupBox.setLayout(vbox)

        return groupBox

    def createSecondGroup(self):
        groupBox = QGroupBox()

        radioLabel = QLabel()
        radioLabel.setAlignment(Qt.AlignLeft)
        # radioLabel.setFont(QFont('Arial', 20))
        radioLabel.setText("Mode de calcul")

        self.radioCalcDP.setChecked(True)

        vbox = QVBoxLayout()
        vbox.addWidget(radioLabel)
        vbox.addWidget(self.radioCalcAll)
        vbox.addWidget(self.radioCalcDP)

        groupBox.setLayout(vbox)

        return groupBox

    def createThirdGroup(self):
        groupBox = QGroupBox()

        radioLabel = QLabel()
        radioLabel.setAlignment(Qt.AlignLeft)
        # radioLabel.setFont(QFont('Arial', 20))
        radioLabel.setText("Mode d'exécution")
        self.radioExecTree.setChecked(True)

        vbox = QVBoxLayout()
        vbox.addWidget(radioLabel)
        vbox.addWidget(self.radioExecStepByStep)
        vbox.addWidget(self.radioExecTree)

        groupBox.setLayout(vbox)

        return groupBox
