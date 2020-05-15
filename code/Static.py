# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:53:50 2020

@author: arian
"""

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class Static(QWidget):
    def __init__(self, parent = None):
        QWidget.__init__(self, parent)

        self.layout = QVBoxLayout(self)

        self.title = QLabel()    
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setWordWrap(True)
        self.title.setFont(QFont('Arial', 20))

        self.list = QListWidget(self)
        self.list.setViewMode(0)

        self.finButton = QPushButton("Fin")

        self.layout.addWidget(self.title)
        self.layout.addWidget(self.list)
        self.layout.addWidget(self.finButton)
        

    def showSequence(self, sequence):
        for i, item in enumerate(sequence):
            self.list.addItem(QListWidgetItem(item))