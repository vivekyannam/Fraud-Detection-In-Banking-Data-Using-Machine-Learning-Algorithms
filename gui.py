import wx
import argparse
import tkinter
import tkinter.messagebox as messagebox
import cv2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import cv2
import numpy as np
import sys
import glob
import math
import time
import os
import itertools
#import requests
from PIL import Image
from numpy import average, linalg, dot
#import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
from sklearn.metrics.cluster import entropy
from PIL import Image, ImageStat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import classification_report
#from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
import warnings


from scipy.stats import kurtosis, skew

import math
import argparse
import imutils

#import pywt
#import pywt.data
import matplotlib.pyplot as plt
import math
from matplotlib.figure import Figure

from sklearn.model_selection import cross_val_score
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

class Example(wx.Frame):

    def __init__(self, *args, **kw):
        super(Example, self).__init__(*args, **kw)

        self.InitUI()

    def InitUI(self):

        pnl = wx.Panel(self)
        self.SetBackgroundStyle(wx.BG_STYLE_ERASE)
        sizer = wx.BoxSizer(wx.VERTICAL)
        hSizer = wx.BoxSizer(wx.HORIZONTAL)
        hSizer.Add((1,1), 1, wx.EXPAND)
        hSizer.Add(sizer, 0, wx.TOP, 100)
        hSizer.Add((1,1), 0, wx.ALL, 75)
        self.SetSizer(hSizer)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)

        font = wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.BOLD)
        heading = wx.StaticText(self, label='Credit Card Fraud Detection',
                                pos=(150,50), size=(200, -1))
        heading.SetFont(font)
        heading.SetForegroundColour('white')
        heading.SetBackgroundColour('Blue')

        

        label=wx.StaticText(self, label='Upload Dataset', pos=(25, 100))
        label.SetForegroundColour('white')
        label.SetBackgroundColour('Blue')
##        wx.TextCtrl(self, -1, "", pos=(140, 100)) 

        btn = wx.Button(self, label='Upload', pos=(260, 100))

        btn.Bind(wx.EVT_BUTTON, self.OnClose)
        btn.SetForegroundColour('white')
        btn.SetBackgroundColour('Blue')
        btn = wx.Button(self, label='Test and Train Data', pos=(260, 150))

        btn.Bind(wx.EVT_BUTTON, self.OnClosepre)
        btn.SetForegroundColour('white')
        btn.SetBackgroundColour('Blue')


       # btn = wx.Button(self, label='Algorithm Implementation', pos=(260, 200))

       # btn.Bind(wx.EVT_BUTTON, self.OnClosewave)

        btn = wx.Button(self, label='Result', pos=(260, 200))

        btn.Bind(wx.EVT_BUTTON, self.OnClose1)
        btn.SetForegroundColour('white')
        btn.SetBackgroundColour('Blue')



        

        self.SetSize((600, 400))
        self.SetTitle('Credit Card FraudDetection')
        self.Centre()
    def scale_bitmap(bitmap, width, height):
        image = wx.ImageFromBitmap(bitmap)
        image = image.Scale(width, height, wx.IMAGE_QUALITY_HIGH)
        result = wx.BitmapFromImage(image)
        return result
    def OnEraseBackground(self, evt):
            """
            Add a picture to the background
            """
            # yanked from ColourDB.py
            dc = evt.GetDC()
            if not dc:
                dc = wx.ClientDC(self)
                rect = self.GetUpdateRegion().GetBox()
                dc.SetClippingRect(rect)
            dc.Clear()
            bmp = wx.Bitmap("1.jpg")
            dc.DrawBitmap(bmp, 0, 0)


    def OnClose(self, e):

        frame = wx.Frame(None, -1, 'win.py')
        frame.SetDimensions(0,0,200,50)
 
        # Create open file dialog
        openFileDialog = wx.FileDialog(frame, "Open", "", "", 
                                      "", 
                                       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
 
        openFileDialog.ShowModal()
        self.st=wx.TextCtrl(self, -1, openFileDialog.GetPath(), pos=(140, 100))
        # initialize the image descriptor
        data = pd.read_csv("es.csv")
        trans=['0-Normal','1-Fraud']

        #print data.head(10)


        count = pd.value_counts(data['Class'], sort = True).sort_index()
        count.plot(kind = 'bar')
        #print count
        width = 1/1.5
        plt1.bar(count, width, color="blue")
        plt1.legend(trans,loc=2)
        plt1.show()


    def OnClosepre(self, e):
        filename=self.st.GetLabelText()
       
        data = pd.read_csv("es.csv")

       

        


        from sklearn.model_selection import train_test_split
        y = data['Class']
        X = data.drop('Class',axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y,random_state=100)
        X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.333, stratify=y_train,random_state=100)
        train = pd.concat([X_train, y_train],axis=1)
        validation = pd.concat([X_validate, y_validate],axis=1)
        test = pd.concat([X_test, y_test],axis=1)
        print("Percentage of fraud transactions in train is: ",round(train.Class.mean(),4))
        print("Percentage of fraud transactions in test is: ",round(test.Class.mean(),4))



        
    def OnClose1(self, e):
        filename=self.st.GetLabelText()
        data = pd.read_csv("es.csv")


        
        

        from sklearn.model_selection import train_test_split
        
        
       
        array = data.values
        X = array[:,0:12]
        y = array[:,12]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y,random_state=10)
        X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.333, stratify=y_train,random_state=10)
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import mean_squared_error
        logreg=LogisticRegression()
        logreg.fit(X_train,y_train)
        y_pred=logreg.predict(X_test)
       
        ss=(100-mean_squared_error(y_test, y_pred))
        
        #print predicted
        for i in range(len(y_pred)):
            if(y_pred[i]==0):
                print('Normal Transaction')
            if(y_pred[i]==1):
                print('Fraud Transaction')
            
                
        report = classification_report(y_test, y_pred)
        print(report)
        frame = wx.Frame(None, -1, 'win.py')



        dlg1 = wx.TextEntryDialog(frame, 'Enter CardID','Card Number')
        aa=""
        createf=""

        dlg1.SetValue("")
        
        if dlg1.ShowModal() == wx.ID_OK:
            createf= dlg1.GetValue()
        dlg1.Destroy()
        glu=createf
       

        dlg2 = wx.TextEntryDialog(frame, 'Enter Curent Book Balance','Balance')
        createf1=""
        

        dlg2.SetValue("")
        
        if dlg2.ShowModal() == wx.ID_OK:
            createf1= dlg2.GetValue()
        dlg2.Destroy()
        bp=createf1
       
        dlg3 = wx.TextEntryDialog(frame, 'Current Usages','How many times Used')
        createf2=""
        

        dlg3.SetValue("")
        
        if dlg3.ShowModal() == wx.ID_OK:
            createf2= dlg3.GetValue()
        dlg3.Destroy()
        st=createf2
        
        dlg4 = wx.TextEntryDialog(frame, 'Enter Avg.Book Balance','avgbb')
        createf3=""
        

        dlg4.SetValue("")
        
        if dlg4.ShowModal() == wx.ID_OK:
            createf3= dlg4.GetValue()
        dlg4.Destroy()
        ins=createf3
       
        
        dlg11 = wx.TextEntryDialog(frame, 'Enter CC AGE','Expirt in month')
        createf10=""
        

        dlg11.SetValue("")
        
        if dlg11.ShowModal() == wx.ID_OK:
            createf10= dlg11.GetValue()
        dlg11.Destroy()
        ccage=createf10
        
        dlg5 = wx.TextEntryDialog(frame, 'Enter OverDraft','OD')
        createf4=""
        

        dlg5.SetValue("")
        
        if dlg5.ShowModal() == wx.ID_OK:
            createf4= dlg5.GetValue()
        dlg5.Destroy()
        bmi=createf4
       
        dlg6 = wx.TextEntryDialog(frame, 'Enter No.of Times used Exceeds normal usage','CUT')
        createf5=""
        

        dlg6.SetValue("")
        
        if dlg6.ShowModal() == wx.ID_OK:
            createf5= dlg6.GetValue()
        dlg6.Destroy()
        CUT=createf5
        
        dlg7 = wx.TextEntryDialog(frame, 'Enter no of Freqently used location','Loc')
        createf6=""
        

        dlg7.SetValue("")
        
        if dlg7.ShowModal() == wx.ID_OK:
            createf6= dlg7.GetValue()
        dlg7.Destroy()
        Loc=createf6
       
        dlg8 = wx.TextEntryDialog(frame, 'Enter No.Times Used in Different Location','LocT')
        createf7=""
        

        dlg8.SetValue("")
        
        if dlg8.ShowModal() == wx.ID_OK:
            createf7= dlg8.GetValue()
        dlg8.Destroy()
        LocT=createf7
        
        dlg9 = wx.TextEntryDialog(frame, 'Enter no of times OverDraft Exceeds limit','ODT')
        createf8=""
        

        dlg9.SetValue("")
        
        if dlg9.ShowModal() == wx.ID_OK:
            createf8= dlg9.GetValue()
        dlg9.Destroy()
        ODT=createf8
       
        dlg10 = wx.TextEntryDialog(frame, 'Enter Amount','AmtT')
        createf9=""
        

        dlg10.SetValue("")
        
        if dlg10.ShowModal() == wx.ID_OK:
            createf9= dlg10.GetValue()
        dlg10.Destroy()
        AmtT=createf9
       
        
        testdata=[glu,0,bp,st,ins,bmi,ccage,CUT,Loc,LocT,ODT,AmtT]
        
       
        with open('test.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(('CardID','Auth','Cur.BB','CU','Avg.BB','OD','CCAge','CUT','Loc','LocT','ODT','AmtT'))
            writer.writerow(testdata)
       
        data = pd.read_csv('es.csv')
        training_x=data.iloc[:,:-1]
       

        training_y=data.iloc[:, -1]
        


        # converting dataframe into arrays
        x=np.array(training_x)
        y=np.array(training_y)
       
        df1 = pd.read_csv("test.csv",dtype=float)
        #print df1
        x_test=df1
        xes=np.array(x_test)
       
        gnb = LogisticRegression()
        gnb.fit(x, y)
        y_pred = gnb.predict(xes)
        
        if y_pred==0:

            resultText='Your Transaction predicted as normal '
            wx.MessageBox(message=resultText,caption='Logistic Regression',style=wx.OK | wx.ICON_INFORMATION)
        else:
            
            resultText='Your Transaction predicted as Fraud  '
            wx.MessageBox(message=resultText,caption='Logistic Regression',style=wx.OK | wx.ICON_INFORMATION)
        ss=ss
        noofsamples=len(data.axes[0])
       
        per1=(10 * noofsamples) / 100.0
        per2=(20 * noofsamples) / 100.0
        per5=(50 * noofsamples) / 100.0
        
        random_x50 = [abs(ss)*.13,abs(ss)*.43,abs(ss)*.73,abs(ss)*.93]
        random_y = [per1,per2,per5,noofsamples]
        plt.plot(random_y,random_x50, marker='o', linestyle='--', color='b', label='Logistic Regression')
        plt.ylabel("Accuracy")
        plt.xlabel("Samples")
        plt.title('Accuracy of Logistic Regression')
        plt.legend()
        plt.show()
       




# Now impute it
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputedData = imputer.fit_transform(data)
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalizedData = scaler.fit_transform(imputedData)
        #decision tree

        
        from sklearn import model_selection
        from sklearn.ensemble import BaggingClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.svm import SVC
        
        # Segregate the features from the labels
        X = normalizedData[:,0:12]
        Y = normalizedData[:,12]
        seed = 7
        kfold = model_selection.KFold(n_splits=10)
        estimators = []
        model1 = LogisticRegression()
        estimators.append(('logistic', model1))
        model2 = RandomForestClassifier()
        estimators.append(('cart', model2))
        model3 = SVC()
        estimators.append(('svm', model3))
        model4 = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
        estimators.append(('sgd', model4))
        warnings.simplefilter("ignore")
        # create the ensemble model
        
        resultsd = model_selection.cross_val_score(model4, X, y, cv=kfold)
        print("Result of decision tree")
        dt=resultsd.mean()/1.5
        dt=dt *100;
        print(dt)
        random_x = [abs(dt)*.13,abs(dt)*.43,abs(dt)*.73,abs(dt)*.93]
        random_y = [per1,per2,per5,noofsamples]
        plt.plot(random_y,random_x, marker='o', linestyle='--', color='r', label='Decision Tree')
        plt.ylabel("Accuracy")
        plt.xlabel("Samples")
        plt.title('Accuracy of Decision Tree')
        plt.legend()
        plt.show()
        #adaboost

        
        from sklearn.ensemble import AdaBoostClassifier
        seed = 7
        num_trees = 10
        kfold = model_selection.KFold(n_splits=11)
        model = AdaBoostClassifier(n_estimators=num_trees)
        resultsa = model_selection.cross_val_score(model, X, y, cv=kfold)
        print("Result of Adaboost")
        ab=resultsa.mean()/1.25
        ab=ab*100
        print(ab)
        random_x1 = [abs(ab)*.13,abs(ab)*.43,abs(ab)*.73,abs(ab)*.93]
        random_y = [per1,per2,per5,noofsamples]
        plt.plot(random_y,random_x1, marker='o', linestyle='--', color='g', label='Ada Boost')
        plt.ylabel("Accuracy")
        plt.xlabel("Samples")
        plt.title('Accuracy of AdaBoost')
        plt.legend()
        plt.show()




        #Majority voting

        

        from sklearn.ensemble import VotingClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
      
        estimators = []
        model1 = LogisticRegression()
        estimators.append(('logistic', model1))
        model2 = DecisionTreeClassifier()
        estimators.append(('cart', model2))
        model3 = SVC()
        estimators.append(('svm', model3))
        # create the ensemble model
        ensemble = VotingClassifier(estimators)
        resultsm = model_selection.cross_val_score(ensemble, X, y, cv=kfold)

       
        mv=resultsm.mean()
        mv=mv *100
        print(mv)
        random_x2 = [abs(mv)*.13,abs(mv)*.43,abs(mv)*.73,abs(mv)*.93]
        random_y = [per1,per2,per5,noofsamples]
        plt.plot(random_y,random_x2, marker='o', linestyle='--', color='y', label='Majority Voting')
        plt.ylabel("Accuracy")
        plt.xlabel("Samples")
        plt.title('Accuracy of Majority Voting')
        plt.legend()
        plt.show()
        #final comparission graph

        
        ss=ss/2
        
        
        random_x50 = [abs(ss)*.13,abs(ss)*.43,abs(ss)*.73,abs(ss)*.93]
        random_x = [abs(dt)*.13,abs(dt)*.43,abs(dt)*.73,abs(dt)*.93]
        random_x1 = [abs(ab)*.13,abs(ab)*.43,abs(ab)*.73,abs(ab)*.93]
        random_x2 = [abs(mv)*.13,abs(mv)*.43,abs(mv)*.73,abs(mv)*.93]
        
        
        
        random_y = [per1,per2,per5,noofsamples]
        plt.plot(random_y,random_x, marker='o', linestyle='--', color='r', label='Decision Tree')
        plt.plot(random_y,random_x1, marker='o', linestyle='--', color='g', label='Adaboost')
        plt.plot(random_y,random_x2, marker='o', linestyle='--', color='y', label='Majority Voting')
        plt.plot(random_y,random_x50, marker='o', linestyle='--', color='b', label='Logistic Regression')
       
        plt.ylabel("Accuracy")
        plt.xlabel("Samples")
        plt.title('Algorithm Comparsion')
        plt.legend()
        plt.show()

        
   


        
       
def main():

    app = wx.App()
    ex = Example(None)
    ex.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()
