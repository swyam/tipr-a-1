# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 22:37:42 2019

@author: swyam
"""


import argparse
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
from numpy import genfromtxt
import operator
import csv
import copy
import random
#import panda as pd
from io import StringIO
from numpy import array
#import imutils
import cv2
# import the necessary packages
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

def randpro(n,y):
    
    Y = np.random.normal(0, 1, (n, y))
    return Y
def knnalgo(datase,size,split):
           

                    trainingSet=[]
                    testSet=[] 
                    def loadDataset(datase, split, trainingSet=[] , testSet=[]):
                            '''with open(filename, 'rb') as csvfile:
                            lines = csv.reader(csvfile)
                            #for row in lines:
                                #print(','.join(row))'''
                            dataset = datase
                            #print(len(lines))
                            for x in range(len(dataset)-1):
                                for y in range(4):
                                    dataset[x][y] = float(dataset[x][y])
                                if random.random() < split:
                                    trainingSet.append(dataset[x])
                                else:
                                    testSet.append(dataset[x])
                     
                     
                    def euclideanDistance(instance1, instance2, length):
                        distance = 0
                        for x in range(length):
                            distance += ((float(instance1[x]) - float(instance2[x]))*(float(instance1[x]) - float(instance2[x])))
                        return np.sqrt(distance)
                     
                    def getNeighbors(trainingSet, testInstance, k):
                        distances = []
                        length = len(testInstance)-1
                        for x in range(len(trainingSet)):
                            dist = euclideanDistance(testInstance, trainingSet[x], length)
                            distances.append((trainingSet[x], dist))
                        distances.sort(key=operator.itemgetter(1))
                        neighbors = []
                        for x in range(k):
                            neighbors.append(distances[x][0])
                        return neighbors
                     
                    def getResponse(neighbors):
                        classVotes = {}
                        for x in range(len(neighbors)):
                            zzz = neighbors[x][-1]
                            #print(zzz)
                            if zzz in classVotes:
                                classVotes[zzz] =classVotes[zzz]+ 1
                            else:
                                classVotes[zzz] = 1
                        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
                        return sortedVotes[0][0]
                     
                    def getAccuracy(testSet, predictions):
                        correct = 0
                        for x in range(len(testSet)):
                            if testSet[x][-1] == predictions[x]:
                                correct += 1
                        return (correct/float(len(testSet))) * 100.0
                    loadDataset(datase, split, trainingSet , testSet)
                    #print ('Train set:' + repr(len(trainingSet)))
                    #print ('Test set: ' + repr(len(testSet)))
                    # generate predictions
                    predictions=[]
                    k = size
                    for x in range(len(testSet)):
                        neighbors = getNeighbors(trainingSet, testSet[x], k)
                        result = getResponse(neighbors)
                        predictions.append(result)
                        #print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
                    accuracy = getAccuracy(testSet, predictions)
                    #print('Accuracy: ' + repr(accuracy) + '%')
                    return accuracy
    
def sklcheck(data1,train_x):

                if int((sklearn.__version__).split(".")[1]) < 18:
                    from sklearn.cross_validation import train_test_split
                 
                # otherwise we're using at lease version 0.18
                else:
                    from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(data1,
                                                                    train_x,
                                                                    test_size = 0.2,
                                                                    random_state=0)
                (X_train, valData, y_train, valLabels) = train_test_split(X_train, y_train,test_size=0.05, random_state=84)
                #print(X_train.shape, X_test.shape)
                # initialize the values of k for our k-Nearest Neighbor classifier along with the
                # list of accuracies for each value of k
                kVals = range(1, 30, 2)
                accuracies = []

                #Create a Gaussian Classifier

                # loop over various values of `k` for the k-Nearest Neighbor classifier
                for k in range(1, 30, 2):
                    # train the k-Nearest Neighbor classifier with the current value of `k`
                    model = KNeighborsClassifier(n_neighbors=k)
                    model.fit(X_train, y_train)
                 

                    # evaluate the model and update the accuracies list
                    score = model.score(valData, valLabels)
                    #print("k=%d, accuracy=%.2f%%" % (k, score * 100))
                    accuracies.append(score)
                 
                # find the value of k that has the largest accuracy
                i = int(np.argmax(accuracies))
                #print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],accuracies[i] * 100))
                model = KNeighborsClassifier(n_neighbors=kVals[i])
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                model = KNeighborsClassifier(n_neighbors=kVals[i])
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                 
                # show a final classification report demonstrating the accuracy of the classifier
                # for each of the digits
                #print("EVALUATION ON TESTING DATA")
                #print(classification_report(y_test, predictions))
                from sklearn import metrics

                # Model Accuracy, how often is the classifier correct?
                #print("F1 score,macro and micro for sklearn knn")
                #accuracy=metrics.accuracy_score(y_test, predictions)
                return classification_report(y_test, predictions)




def Bayes(datase,size,split):
           
                    dataset=datase
                    trainingSet=[]
                    testSet=[] 
                    def loadDataset(datase, split, trainingSet=[] , testSet=[]):
                            '''with open(filename, 'rb') as csvfile:
                            lines = csv.reader(csvfile)
                            #for row in lines:
                                #print(','.join(row))'''
                            dataset = datase
                            #print(len(lines))
                            for x in range(len(dataset)-1):
                                for y in range(4):
                                    dataset[x][y] = float(dataset[x][y])
                                if random.random() < split:
                                    trainingSet.append(dataset[x])
                                else:
                                    testSet.append(dataset[x])
                            #print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
                     

                    def separateByClass(dataset):
                        separated = {}
                        for i in range(len(dataset)):
                            #vector = dataset[i]
                            if (dataset[i][-1] not in separated):
                                separated[dataset[i][-1]] = []
                            separated[dataset[i][-1]].append(dataset[i])
                        return separated
                   
                    
                    def mean(arr):
                        length=len(arr)
                        sum=0
                        for i in range(0,length):
                            sum=sum+arr[i]
                        sum=float(sum)/float(length)
                        return sum
                    def stdv(arr):
                        length=len(arr)
                        #print(length)
                        un=mean(arr)
                        sum=0
                        for i in range(0,length):
                            sum=sum+(arr[i]-un)*(arr[i]-un)
                        sum=float(sum)/float(length)
                        return np.sqrt(sum)
                    def summarize(dataset):
                        
                        summaries = [(mean(attribute), stdv(attribute)) for attribute in zip(*dataset)]
                        del summaries[-1]
                        return summaries
                    def sumbyclass(dataset):
                        separated=separateByClass(dataset)
                        summaries = {}
                        for classValue, instances in separated.items():
                              summaries[classValue] = summarize(instances)
                        return summaries
                    def calculateProbability(x, mean, stdev):
                        exponent = np.exp(-(float(x-mean)*(x-mean))/(2*(float(stdev*stdev+0.0000001))))
                        #print (stdev)
                        return (1 / float(np.sqrt(2*np.pi) * (stdev+.000001)) * exponent)
                    
                    
                    def calculateClassProbabilities(summaries, inputVector):
                            probabilities = {}
                            for classValue, classSummaries in summaries.items():
                                probabilities[classValue] = 1
                                for i in range(len(classSummaries)):
                                    mean, stdev = classSummaries[i]
                                    x = inputVector[i]
                                    probabilities[classValue] *= calculateProbability(x, mean, stdev)
                            return probabilities
                    
                    
                    def predict(summaries, inputVector):
                                    probabilities = calculateClassProbabilities(summaries, inputVector)
                                    bestLabel, bestProb = None, -1
                                    for classValue, probability in probabilities.items():
                                        if bestLabel is None or probability > bestProb:
                                            bestProb = probability
                                            bestLabel = classValue
                                    return bestLabel
                   
                    
                    def getPredictions(summaries, testSet):
                                    predictions = []
                                    for i in range(len(testSet)):
                                        result = predict(summaries, testSet[i])
                                        predictions.append(result)
                                    return predictions
                    
                   
                    
                    def getAccuracy(testSet, predictions):
                                    correct = 0
                                    for x in range(len(testSet)):
                                        #print("predicted",predictions[x]," ","actual",testSet[x][-1]," ")
                                        if testSet[x][-1] == predictions[x]:
                                            
                                            correct += 1
                                            #print(len(testSet))
                                    return (correct/float(len(testSet))) * 100.0
                   
                    loadDataset(datase,split,trainingSet,testSet)
                    
                   
                    summaries = sumbyclass(trainingSet)
                    #result = predict(summaries, inputVector)
                    #print('Prediction: {0}'.format(result))
                    predictions = getPredictions(summaries, testSet)
                    accuracy = getAccuracy(testSet, predictions)
                    accuracy=getAccuracy(testSet, predictions)
                    #print('bayes Accuracy: {0}'.format(accuracy))
                    return accuracy;

def skbay(aaa1,train_x):
           #Create a Gaussian Classifier
            gnb = GaussianNB()
            if int((sklearn.__version__).split(".")[1]) < 18:
                from sklearn.cross_validation import train_test_split
                 
                
            else:
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(aaa1,
                                                                    train_x,
                                                                    test_size = 0.2,
                                                                    random_state=0)
                (X_train, valData, y_train, valLabels) = train_test_split(X_train, y_train,test_size=0.05, random_state=84)
                #print(X_train.shape, X_test.shape)

            #Train the model using the training sets
            gnb.fit(X_train, y_train)

            #Predict the response for test dataset
            y_pred = gnb.predict(X_test)
            
            #print(y_pred)

            #from sklearn import metrics
            #from sklearn.metrics import f1_score

            # Model Accuracy, how often is the classifier correct?
            #print("F1 score,macro and micro for sklearn bayes")
            #print(f1_score(y_test, y_pred, average='macro') )
            accuracy=classification_report(y_test, y_pred)
            
            return accuracy
                                
            


##########################################################




data_x=np.genfromtxt('../data/dolphins/dolphins.csv',delimiter=' ')
train_x=np.genfromtxt('../data/dolphins/dolphins_label.csv',delimiter=' ')
#print(data_x)



m=len(data_x)
n=len(data_x[0])
print(m," ",n)


a=4
b=n
step=20

r=(b/a)**(1/(step-1))
for xxt in range(0,step):

    k=a*r**xxt
    k=int(round(k))

    Y=randpro(n,k)
    data1 =np.matmul(data_x,Y)
    
    aaa1=np.squeeze(np.asarray(np.copy(data1)))
    
    aaa=np.column_stack([np.copy(aaa1),np.copy(train_x)])
    #print (aaa)
    print("dimensions",k)
    accuknn= knnalgo(np.copy(aaa),8,0.80)
    print("Accuracy KNN ALGO",accuknn)
    
    
    accubayes=  Bayes(np.copy(aaa),10,0.80)
    print("Accuracy Bayes ALGO",accubayes)
    
    accsknn=sklcheck(np.copy(aaa1),train_x)
    print("Accuracy sklearn  KNN",accsknn)
    
    accskb=skbay(np.copy(aaa1),train_x)
    print("Accuracy sklearn Bayes",accskb)
 
    print("")
    
        
    
    #print(accu)
