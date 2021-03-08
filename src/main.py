# main.py
#
# This class creates a GUI for input data.
#
# Created by: Constandinos Demetriou, Mar 2021

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMessageBox
import sys
import classes.prediction as prediction


class Ui_Form(object):

    def __init__(self):
        self.model = prediction.load_model()

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1225, 966)
        icon = QtGui.QIcon()
        # window icon source: https://en.wikipedia.org/wiki/Heart_symbol
        icon.addPixmap(QtGui.QPixmap("icons/window_icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Form.setWindowIcon(icon)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.label_header = QtWidgets.QLabel(Form)
        self.label_header.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_header.setText("")
        # header image source: https://www.diabetes.co.uk/diabetes-complications/preventing-heart-disease.html
        self.label_header.setPixmap(QtGui.QPixmap("icons/header.bmp"))
        self.label_header.setAlignment(QtCore.Qt.AlignCenter)
        self.label_header.setObjectName("label_header")
        self.gridLayout.addWidget(self.label_header, 0, 0, 1, 1)
        self.label_title = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setFamily("Script MT Bold")
        font.setPointSize(26)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setAlignment(QtCore.Qt.AlignCenter)
        self.label_title.setObjectName("label_title")
        self.gridLayout.addWidget(self.label_title, 1, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_age = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_age.setFont(font)
        self.label_age.setObjectName("label_age")
        self.verticalLayout.addWidget(self.label_age)
        self.label_sex = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_sex.setFont(font)
        self.label_sex.setObjectName("label_sex")
        self.verticalLayout.addWidget(self.label_sex)
        self.label_cp = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_cp.setFont(font)
        self.label_cp.setObjectName("label_cp")
        self.verticalLayout.addWidget(self.label_cp)
        self.label_trestbps = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_trestbps.setFont(font)
        self.label_trestbps.setObjectName("label_trestbps")
        self.verticalLayout.addWidget(self.label_trestbps)
        self.label_chol = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_chol.setFont(font)
        self.label_chol.setObjectName("label_chol")
        self.verticalLayout.addWidget(self.label_chol)
        self.label_fbs = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_fbs.setFont(font)
        self.label_fbs.setObjectName("label_fbs")
        self.verticalLayout.addWidget(self.label_fbs)
        self.label_restecg = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_restecg.setFont(font)
        self.label_restecg.setObjectName("label_restecg")
        self.verticalLayout.addWidget(self.label_restecg)
        self.label_thalach = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_thalach.setFont(font)
        self.label_thalach.setObjectName("label_thalach")
        self.verticalLayout.addWidget(self.label_thalach)
        self.label_exang = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_exang.setFont(font)
        self.label_exang.setObjectName("label_exang")
        self.verticalLayout.addWidget(self.label_exang)
        self.label_oldpeak = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_oldpeak.setFont(font)
        self.label_oldpeak.setObjectName("label_oldpeak")
        self.verticalLayout.addWidget(self.label_oldpeak)
        self.label_slope = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_slope.setFont(font)
        self.label_slope.setObjectName("label_slope")
        self.verticalLayout.addWidget(self.label_slope)
        self.label_ca = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_ca.setFont(font)
        self.label_ca.setObjectName("label_ca")
        self.verticalLayout.addWidget(self.label_ca)
        self.label_thal = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_thal.setFont(font)
        self.label_thal.setObjectName("label_thal")
        self.verticalLayout.addWidget(self.label_thal)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.lineEdit_age = QtWidgets.QLineEdit(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        self.lineEdit_age.setFont(font)
        self.lineEdit_age.setObjectName("lineEdit_age")
        self.verticalLayout_2.addWidget(self.lineEdit_age)
        self.comboBox_sex = QtWidgets.QComboBox(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        self.comboBox_sex.setFont(font)
        self.comboBox_sex.setObjectName("comboBox_sex")
        self.comboBox_sex.addItem("")
        self.comboBox_sex.addItem("")
        self.verticalLayout_2.addWidget(self.comboBox_sex)
        self.comboBox_cp = QtWidgets.QComboBox(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        self.comboBox_cp.setFont(font)
        self.comboBox_cp.setObjectName("comboBox_cp")
        self.comboBox_cp.addItem("")
        self.comboBox_cp.addItem("")
        self.comboBox_cp.addItem("")
        self.comboBox_cp.addItem("")
        self.verticalLayout_2.addWidget(self.comboBox_cp)
        self.lineEdit_trestbps = QtWidgets.QLineEdit(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        self.lineEdit_trestbps.setFont(font)
        self.lineEdit_trestbps.setObjectName("lineEdit_trestbps")
        self.verticalLayout_2.addWidget(self.lineEdit_trestbps)
        self.lineEdit_chol = QtWidgets.QLineEdit(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        self.lineEdit_chol.setFont(font)
        self.lineEdit_chol.setObjectName("lineEdit_chol")
        self.verticalLayout_2.addWidget(self.lineEdit_chol)
        self.comboBox_fbs = QtWidgets.QComboBox(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        self.comboBox_fbs.setFont(font)
        self.comboBox_fbs.setObjectName("comboBox_fbs")
        self.comboBox_fbs.addItem("")
        self.comboBox_fbs.addItem("")
        self.verticalLayout_2.addWidget(self.comboBox_fbs)
        self.comboBox_restecg = QtWidgets.QComboBox(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        self.comboBox_restecg.setFont(font)
        self.comboBox_restecg.setObjectName("comboBox_restecg")
        self.comboBox_restecg.addItem("")
        self.comboBox_restecg.addItem("")
        self.comboBox_restecg.addItem("")
        self.verticalLayout_2.addWidget(self.comboBox_restecg)
        self.lineEdit_thalach = QtWidgets.QLineEdit(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        self.lineEdit_thalach.setFont(font)
        self.lineEdit_thalach.setObjectName("lineEdit_thalach")
        self.verticalLayout_2.addWidget(self.lineEdit_thalach)
        self.comboBox_exang = QtWidgets.QComboBox(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        self.comboBox_exang.setFont(font)
        self.comboBox_exang.setObjectName("comboBox_exang")
        self.comboBox_exang.addItem("")
        self.comboBox_exang.addItem("")
        self.verticalLayout_2.addWidget(self.comboBox_exang)
        self.lineEdit_oldpeak = QtWidgets.QLineEdit(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        self.lineEdit_oldpeak.setFont(font)
        self.lineEdit_oldpeak.setObjectName("lineEdit_oldpeak")
        self.verticalLayout_2.addWidget(self.lineEdit_oldpeak)
        self.comboBox_slope = QtWidgets.QComboBox(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        self.comboBox_slope.setFont(font)
        self.comboBox_slope.setObjectName("comboBox_slope")
        self.comboBox_slope.addItem("")
        self.comboBox_slope.addItem("")
        self.comboBox_slope.addItem("")
        self.verticalLayout_2.addWidget(self.comboBox_slope)
        self.comboBox_ca = QtWidgets.QComboBox(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        self.comboBox_ca.setFont(font)
        self.comboBox_ca.setObjectName("comboBox_ca")
        self.comboBox_ca.addItem("")
        self.comboBox_ca.addItem("")
        self.comboBox_ca.addItem("")
        self.comboBox_ca.addItem("")
        self.verticalLayout_2.addWidget(self.comboBox_ca)
        self.comboBox_thal = QtWidgets.QComboBox(Form)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic")
        font.setPointSize(12)
        self.comboBox_thal.setFont(font)
        self.comboBox_thal.setObjectName("comboBox_thal")
        self.comboBox_thal.addItem("")
        self.comboBox_thal.addItem("")
        self.comboBox_thal.addItem("")
        self.verticalLayout_2.addWidget(self.comboBox_thal)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.gridLayout.addLayout(self.horizontalLayout, 2, 0, 1, 1)
        self.pushButton_submit = QtWidgets.QPushButton(Form)
        font = QtGui.QFont()
        font.setFamily("High Tower Text")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_submit.setFont(font)
        self.pushButton_submit.setObjectName("pushButton_submit")
        self.gridLayout.addWidget(self.pushButton_submit, 3, 0, 1, 1)
        self.pushButton_clear = QtWidgets.QPushButton(Form)
        font = QtGui.QFont()
        font.setFamily("High Tower Text")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_clear.setFont(font)
        self.pushButton_clear.setObjectName("pushButton_clear")
        self.gridLayout.addWidget(self.pushButton_clear, 4, 0, 1, 1)
        self.label_info = QtWidgets.QLabel(Form)
        self.label_info.setAlignment(QtCore.Qt.AlignCenter)
        self.label_info.setObjectName("label_info")
        self.gridLayout.addWidget(self.label_info, 5, 0, 1, 1)

        self.retranslateUi(Form)
        self.pushButton_submit.clicked.connect(self.submitSlot)
        self.pushButton_clear.clicked.connect(self.clearSlot)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Prediction of heart disease"))
        self.label_title.setText(_translate("Form", "Prediction of heart disease in a patient"))
        self.label_age.setText(_translate("Form", "Age"))
        self.label_sex.setText(_translate("Form", "Sex"))
        self.label_cp.setText(_translate("Form", "Chest pain type"))
        self.label_trestbps.setText(_translate("Form", "Resting blood pressure (mm Hg)"))
        self.label_chol.setText(_translate("Form", "Serum cholestoral (mg/dl)"))
        self.label_fbs.setText(_translate("Form", "Fasting blood sugar (mg/dl)"))
        self.label_restecg.setText(_translate("Form", "Resting electrocardiographic results"))
        self.label_thalach.setText(_translate("Form", "Maximum heart rate achieved"))
        self.label_exang.setText(_translate("Form", "Exercise induced angina"))
        self.label_oldpeak.setText(_translate("Form", "ST depression induced by exercise relative to rest"))
        self.label_slope.setText(_translate("Form", "The slope of the peak exercise ST segment"))
        self.label_ca.setText(_translate("Form", "Number of major vessels (0-3) colored by flourosopy"))
        self.label_thal.setText(_translate("Form", "Thalassemia"))
        self.comboBox_sex.setItemText(0, _translate("Form", "Female"))
        self.comboBox_sex.setItemText(1, _translate("Form", "Male"))
        self.comboBox_cp.setItemText(0, _translate("Form", "Typical angina"))
        self.comboBox_cp.setItemText(1, _translate("Form", "Atypical angina"))
        self.comboBox_cp.setItemText(2, _translate("Form", "Non-anginal pain"))
        self.comboBox_cp.setItemText(3, _translate("Form", "Asymptomatic"))
        self.comboBox_fbs.setItemText(0, _translate("Form", "<= 120"))
        self.comboBox_fbs.setItemText(1, _translate("Form", "> 120"))
        self.comboBox_restecg.setItemText(0, _translate("Form", "Normal"))
        self.comboBox_restecg.setItemText(1, _translate("Form", "Having ST-T wave abnormality"))
        self.comboBox_restecg.setItemText(2, _translate("Form",
                                                        "Showing probable or definite left ventricular hypertrophy"))
        self.comboBox_exang.setItemText(0, _translate("Form", "No"))
        self.comboBox_exang.setItemText(1, _translate("Form", "Yes"))
        self.comboBox_slope.setItemText(0, _translate("Form", "Upsloping"))
        self.comboBox_slope.setItemText(1, _translate("Form", "Flat"))
        self.comboBox_slope.setItemText(2, _translate("Form", "Downsloping"))
        self.comboBox_ca.setItemText(0, _translate("Form", "0"))
        self.comboBox_ca.setItemText(1, _translate("Form", "1"))
        self.comboBox_ca.setItemText(2, _translate("Form", "2"))
        self.comboBox_ca.setItemText(3, _translate("Form", "3"))
        self.comboBox_thal.setItemText(0, _translate("Form", "Normal"))
        self.comboBox_thal.setItemText(1, _translate("Form", "Fixed defect"))
        self.comboBox_thal.setItemText(2, _translate("Form", "Reversable defect"))
        self.pushButton_submit.setText(_translate("Form", "Submit"))
        self.pushButton_clear.setText(_translate("Form", "Clear"))
        self.label_info.setText(_translate("Form", "University of Cyprus - Department of Computer Science\n"
                                                   "Constandinos Demetriou © 2021"))

    def show_msg(self, msg, title):
        """
        Appears a message in a window in case that an error occurred or a prediction result.

        Parameters
	    ----------
	    msg: str
	        The message that will appeare in the window.
	    title: str
	        The title of message window
        """

        # create message box
        self.msg_box = QMessageBox()
        # warning icon source: https://icons8.com/icon/DHIgPGXMCn0B/error
        self.msg_box.setWindowIcon(QtGui.QIcon('icons/warning.png'))

        # set icons
        if title == 'Error':
            self.msg_box.setIcon(QMessageBox.Critical)
        elif title == 'Prediction normal':
            # happy icon source: https://en.wikipedia.org/wiki/Smiley
            self.msg_box.setIconPixmap(QPixmap('icons/happy.png'))
        elif title == 'Prediction disease':
            # sad icon source: https://www.pinterest.ca/pin/310396599309432868/
            self.msg_box.setIconPixmap(QPixmap('icons/sad.png').scaledToWidth(40))

        # set title and message
        self.msg_box.setWindowTitle(title)
        self.msg_box.setText(msg)

        # execute
        x = self.msg_box.exec_()

    def submitSlot(self):
        """
        Gets all values from form, checks for empty values and makes the prediction.
        """

        # get values
        # age
        age = self.lineEdit_age.text()
        # sex
        if self.comboBox_sex.currentText() == 'Female':
            sex = 0
        elif self.comboBox_sex.currentText() == 'Male':
            sex = 1
        # chest pain type
        if self.comboBox_cp.currentText() == 'Typical angina':
            cp = 0
        elif self.comboBox_cp.currentText() == 'Atypical angina':
            cp = 1
        elif self.comboBox_cp.currentText() == 'Non-anginal pain':
            cp = 2
        elif self.comboBox_cp.currentText() == 'Asymptomatic':
            cp = 3
        # resting blood pressure
        trestbps = self.lineEdit_trestbps.text()
        # serum cholestoral
        chol = self.lineEdit_chol.text()
        # fasting blood sugar
        if self.comboBox_fbs.currentText() == '<= 120':
            fbs = 0
        elif self.comboBox_fbs.currentText() == '> 120':
            fbs = 1

        # resting electrocardiographic results
        if self.comboBox_restecg.currentText() == 'Normal':
            restecg = 0
        elif self.comboBox_restecg.currentText() == 'Having ST-T wave abnormality':
            restecg = 1
        elif self.comboBox_restecg.currentText() == 'Showing probable or definite left ventricular hypertrophy':
            restecg = 2
        # maximum heart rate achieved
        thalach = self.lineEdit_thalach.text()
        # exercise induced angina
        if self.comboBox_exang.currentText() == 'No':
            exang = 0
        elif self.comboBox_exang.currentText() == 'Yes':
            exang = 1
        # ST depression induced by exercise relative to rest
        oldpeak = self.lineEdit_oldpeak.text()
        # the slope of the peak exercise ST segment
        if self.comboBox_slope.currentText() == 'Upsloping':
            slope = 0
        elif self.comboBox_slope.currentText() == 'Flat':
            slope = 1
        elif self.comboBox_slope.currentText() == 'Downsloping':
            slope = 2
        # number of major vessels (0-3) colored by flourosopy
        ca = int(self.comboBox_ca.currentText())
        # thalassemia
        if self.comboBox_thal.currentText() == 'Normal':
            thal = 0
        elif self.comboBox_thal.currentText() == 'Fixed defect':
            thal = 1
        elif self.comboBox_thal.currentText() == 'Reversable defect':
            thal = 2

        # check for empty values
        if age == '' or trestbps == '' or chol == '' or thalach == '' or oldpeak == '':
            self.show_msg('Error: Some fields are empty. Please fill all values!', 'Error')
            return
        # converse string to int
        else:
            age = int(age)
            trestbps = int(trestbps)
            chol = int(chol)
            thalach = int(thalach)
            oldpeak = float(oldpeak)

        # check for negative values
        if age < 0 or trestbps < 0 or chol < 0 or thalach < 0 or oldpeak < 0:
            self.show_msg('Error: Some values are negative. Please give only positive values!', 'Error')
            return
        elif age > 100:
            self.show_msg('Error: The system works only for patient with age under 100 years old. Please '
                          'values for patients with age under 100 years old!', 'Error')
            return

        # make prediction
        pred = prediction.make_prediction(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope,
                                          ca, thal, self.model)

        # appeare prediction results
        if pred == 0:
            self.show_msg('Good news! The system predicts that the patient has not any heart disease.', 'Prediction '
                                                                                                        'normal')
        elif pred == 1:
            self.show_msg('Bad news. The system predicts that the patient has a heart disease. Please contact with '
                          'your doctor immediately!', 'Prediction disease')

    def clearSlot(self):
        """
        Cleares all line edits and resets combobox.
        """
        # clear line edits
        self.lineEdit_age.setText('')
        self.lineEdit_trestbps.setText('')
        self.lineEdit_chol.setText('')
        self.lineEdit_thalach.setText('')
        self.lineEdit_oldpeak.setText('')

        # reset combobox
        self.comboBox_sex.setCurrentIndex(0)
        self.comboBox_cp.setCurrentIndex(0)
        self.comboBox_fbs.setCurrentIndex(0)
        self.comboBox_restecg.setCurrentIndex(0)
        self.comboBox_exang.setCurrentIndex(0)
        self.comboBox_slope.setCurrentIndex(0)
        self.comboBox_ca.setCurrentIndex(0)
        self.comboBox_thal.setCurrentIndex(0)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(window)
    window.show()
    sys.exit(app.exec_())
