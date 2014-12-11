'''
Program to forecast solare flares using a Support Vector Machine algorithm
This code uses the scikit-learn Python module
Different catalogs of features of flaring solar active regions and non flaring ones
are used depending on the definition of a flare.
The active-region features are taken from the HMI SHARP parameters (Bobra et al., 2014)
'''

import time, numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score, accuracy_score,f1_score,mean_squared_error
from sklearn.preprocessing import LabelBinarizer, scale
from sklearn.decomposition import PCA 
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, SelectFpr, SelectKBest
from sklearn.utils import shuffle

start = time.clock()

#FLARE FEATURE CATALOG
#----------------------------------------------------------------------

with open('flare_catalog_24h.txt','r') as flare_file:   #for features of active regions taken 24h prior to a flare
#with open('flare_catalog_12h.txt','r') as flare_file:  #for features of active regions taken 12h prior to a flare
    Xtext=flare_file.readlines()
flare_file.close()
Xflare=[]

with open('flare_catalog_24h_times_d0_out.txt','r') as flare_file:  #to add the fractal dimension
    Xtext2=flare_file.readlines()
flare_file.close()

for i in range(len(Xtext)):
    res=Xtext[i].split()
    res2=Xtext2[i].split()
    Xflare.append([float(res[0]),float(res[1]),float(res[2]),float(res[3]),float(res[4]),float(res[5]),float(res[6]),float(res[7]),float(res[8]),float(res[9]),float(res[10]),float(res[11]),float(res[12]),float(res[13]),float(res[14]),float(res[15]),float(res[16]),float(res[17]),float(res[18]),float(res[19]),float(res[20]),float(res[21]),float(res[22]),float(res[23]),float(res[24]),float(res2[0])])

#NO-FLARE FEATURE CATALOG
#---------------------------------------------------------------------

Xnoflare=[]
#with open('noflare_catalog_24h.txt','r') as noflare_file:  #for active regions not flaring in a 24h interval prior to a flare 
with open('noflare_catalog_48h.txt','r') as noflare_file:   #for active regions not flaring in a 48h interval around a flare
    Xtext=noflare_file.readlines()
noflare_file.close()

with open('noflare_catalog_48h_times_d0_out.txt','r') as noflare_file:  #to add the fractal dimension
    Xtext2=noflare_file.readlines()
noflare_file.close()

for i in range(len(Xtext)):
    res=Xtext[i].split()
    res2=Xtext2[i].split()
    Xnoflare.append([float(res[0]),float(res[1]),float(res[2]),float(res[3]),float(res[4]),float(res[5]),float(res[6]),float(res[7]),float(res[8]),float(res[9]),float(res[10]),float(res[11]),float(res[12]),float(res[13]),float(res[14]),float(res[15]),float(res[16]),float(res[17]),float(res[18]),float(res[19]),float(res[20]),float(res[21]),float(res[22]),float(res[23]),float(res[24]),float(res2[0])])

Xflare = np.array(Xflare,dtype=np.float64)
Xnoflare = np.array(Xnoflare,dtype=np.float64)
#Xflare = np.delete(Xflare,[1,2,6,7,8,9,10,14,17,18,21,22],1) #best features according to univariate F-score ranking
#Xnoflare = np.delete(Xnoflare,[1,2,6,7,8,9,10,14,17,18,21,22],1)

#Xflare = np.delete(Xflare,[0,1,3,4,5,8,11,13,14,15,19,20,21],1) #best features according to mRMR difference
#Xnoflare = np.delete(Xnoflare,[0,1,3,4,5,8,11,13,14,15,19,20,21],1)

#Xflare = np.delete(Xflare,[1,2,4,6,7,8,9,14,15,17,19,21,22],1) #best features according to mRMR quotient
#Xnoflare = np.delete(Xnoflare,[1,2,4,6,7,8,9,14,15,17,19,21,22],1)

#Xflare = np.delete(Xflare,[0,1,2,4,6,7,8,9,14,17,19,21,22],1) #best features according to mRMR quotient with non-dominated condition
#Xnoflare = np.delete(Xnoflare,[0,1,2,4,6,7,8,9,14,17,19,21,22],1)

Xflare = np.delete(Xflare,[0,1,3,4,5,8,11,13,14,15,19,20,21],1) #best features according to mRMR difference with non-dominated condition
Xnoflare = np.delete(Xnoflare,[0,1,3,4,5,8,11,13,14,15,19,20,21],1)

#Xflare = np.delete(Xflare,[21,22,17,7,10,18,2,6,9,1,8,14,4,23,16,3,19,15,0,12,24,11,13,20],1)
#Xnoflare = np.delete(Xnoflare,[21,22,17,7,10,18,2,6,9,1,8,14,4,23,16,3,19,15,0,12,24,11,13,20],1)
                          
#FOR PIL
#Xflare = np.delete(Xflare,[1,2,3,4,6,7,8,9,10,14,17,18,19,21,22,23],1) #remove features with a low univariate score
#Xnoflare = np.delete(Xnoflare,[1,2,3,4,6,7,8,9,10,14,17,18,19,21,22,23],1) #remove features with a low univariate score

#FEATUE SCALING
#--------------------------------------------------------------------

for i in range(Xflare.shape[1]):
    meancat=np.median(np.append(Xflare[:,i],Xnoflare[:,i]))
    sigmacat=np.std(np.append(Xflare[:,i],Xnoflare[:,i]))
    Xflare[:,i] = (Xflare[:,i]-meancat)/sigmacat
    Xnoflare[:,i]= (Xnoflare[:,i]-meancat)/sigmacat
    #meancat=np.mean(Xnoflare[:,i])
    #sigmacat=np.std(Xnoflare[:,i])
    #Xflare[:,i] = (Xflare[:,i]-meancat)/sigmacat
    #Xnoflare[:,i]= (Xnoflare[:,i]-meancat)/sigmacat
    #meancat=np.mean(np.append(Xflare[:,i],Xnoflare[:,i]))
    #sigmacat=np.amax(np.append(Xflare[:,i],Xnoflare[:,i]))-np.amin(np.append(Xflare[:,i],Xnoflare[:,i]))
    #Xflare[:,i] = (Xflare[:,i]-meancat)/sigmacat
    #Xnoflare[:,i]= (Xnoflare[:,i]-meancat)/sigmacat

#selector = SelectFpr(f_classif, alpha=0.1)
#selector = SelectPercentile(f_classif, percentile=10)
selector = SelectKBest(f_classif,k="all")
selector.fit(np.append(Xflare,Xnoflare,axis=0),np.append(np.ones(Xflare.shape[0]),np.zeros(Xnoflare.shape[0]),axis=0))
scores=selector.scores_
#scores = -np.log10(selector.pvalues_)
#scores /= scores.max()
print 'Univariate scores before feature selection=',scores

#selector.fit(X, y)
#scores = -np.log10(selector.pvalues_)
#scores=selector.scores_
#scores /= scores.max()
#print 'Univariate scores after feature selection=',scores
#print 'confidence levels:',-np.log10(selector.pvalues_)

#SVM ALGORITHM
#-----------------------------------------------------------------

C = 10  # SVM regularization parameter (1 for multivariate score and best TSS, 4 otherwise)
clf   = svm.SVC(kernel='rbf',gamma=0.080,C=C,class_weight={1:2.0},cache_size=400,tol=1e-8) #parameters for best HSS2: gamma=0.075, class_weight={1:2.00}
#clf   = svm.SVC(kernel='rbf',gamma=0.075,C=C,class_weight={1:17.0},cache_size=400,tol=1e-8) #parameters for best TSS with multivariate score
#clf   = svm.SVC(kernel='rbf',gamma=0.075,C=C,class_weight={1:15.0},cache_size=400,tol=1e-8) #parameters for best TSS with univariate score

# TRAINING AND TESTING THE SVM
#-----------------------------------------------------------------

ntest = 400 #number of tests (1 test= select random negative examples to add to the positive ones and create the training set, then preprocess, then train the SVM, then predict on a cross-validation set
#nexamples=14989 #number of negative examples for the testing set
nexamples=5231 #5180 #4991
rec   = np.zeros((ntest,2)) #recall (sensitivity)
prec  = np.zeros((ntest,2)) #precision
acc   = np.zeros(ntest) #accuracy
skill = np.zeros(ntest) #Heidke skill score from Barnes and Leka (2008)
skill2= np.zeros(ntest) #Heidke skill score from Balch (2008) and the Space Weather Prediction Center
f1    = np.zeros((ntest,2)) #F1-score
tp    = np.zeros(ntest) #true positive
tn    = np.zeros(ntest) #true negative
fn    = np.zeros(ntest) #false negative
fp    = np.zeros(ntest) #false positive
E     = np.zeros(ntest) #benchmark for the Skill score defined by Graham and Leka (2008)
fpr   = np.zeros(ntest) #False Positive Rate (fall out)
fnr   = np.zeros(ntest) #False Negative Rate
CH    = np.zeros(ntest) #chance
gilbert=np.zeros(ntest) #Gilbert skill score
MSE   = np.zeros(ntest) #mean squared error
tnr   = np.zeros(ntest) #specificity, true negative rate
TSS   = np.zeros(ntest) #True skill statistics

numtraip=np.zeros(ntest)#number of positive examples in the training set
numtrain=np.zeros(ntest)#number of negatives examples in the training set
numtestp=np.zeros(ntest)#number of positive examples in the testing set
numtestn=np.zeros(ntest)#number of negative examples in the testing set

 
for ii in range(ntest):

    #create training set
    X=np.array(Xflare)            #positive examples
    X2=np.array(Xnoflare)
    np.random.shuffle(X2)         #shuffle catalog of no-flares
    X3=X2[0:nexamples,:]          #sub-sampling to avoid imbalanced training set
    X=np.append(X,X3,axis=0)
    
    y  = np.ones(Xflare.shape[0],dtype=np.int16) #labels have to be integer for the SVC function in case of classification
    y  = np.append(y,-1*np.ones(nexamples,dtype=np.int16))             
    
    '''
    selector = SelectFpr(f_classif, alpha=0.1)
    selector = SelectPercentile(f_classif, percentile=10)
    selector = SelectKBest(f_classif,k="all")
    selector.fit(X, y)
    scores=selector.scores_
    print 'Univariate scores before feature selection=',scores
    input()
    '''

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30) #split the example set into a training and testing set
    clf.fit(X_train, y_train)  #train the SVM on the training set

    #create testing set
    X_test=np.append(X_test,X2[nexamples:,],axis=0)
    y_test=np.append(y_test,-1*np.ones(X2.shape[0]-nexamples,dtype=np.int16))
    

    numtraip[ii]=sum([yo>0 for yo in y_train])
    numtrain[ii]=sum([yo<0 for yo in y_train])
    numtestp[ii]=sum([yo>0 for yo in y_test])
    numtestn[ii]=sum([yo<0 for yo in y_test])

    predictions = []
    predictions = np.array(predictions,dtype=np.float64)

    m=float(X_test.shape[0])
    
    for i in range(X_test.shape[0]):
        o=clf.predict(X_test[i])
        if o <= 0.0:
            predictions=np.append(predictions,-1.0) #negative predictions
        elif o > 0.0:
            predictions=np.append(predictions,1.0)  #positive predictions

    #print classification_report(y_test,predictions)
    tp[ii]=sum(x>1.0 for x in predictions+y_test)
    tn[ii]=sum(x<-1.0 for x in predictions+y_test)
    fn[ii]=sum(x>1.0 for x in y_test-predictions)
    fp[ii]=sum(x<-1.0 for x in y_test-predictions)
    #rec[ii]=tp[ii]/(tp[ii]+fn[ii])  #recall_score(y_test,predictions)
    #prec[ii]=tp[ii]/(tp[ii]+fp[ii]) #precision_score(y_test,predictions)
    rec[ii,:]=recall_score(y_test,predictions,average=None)
    prec[ii,:]=precision_score(y_test,predictions,average=None)
    acc[ii]=(tp[ii]+tn[ii])/(tp[ii]+fn[ii]+tn[ii]+fp[ii])#accuracy_score(y_test,predictions)
    skill[ii]=(tp[ii]-fp[ii])/(tp[ii]+fn[ii])#rec[ii]*(2.0-1./prec[ii])
    E[ii]=((tp[ii]+fp[ii])*(tp[ii]+fn[ii])+(fp[ii]+tn[ii])*(fn[ii]+tn[ii]))/m
    skill2[ii]=(tp[ii]+tn[ii]-E[ii])/(tp[ii]+fp[ii]+fn[ii]+tn[ii]-E[ii])
    f1[ii,:]=f1_score(y_test,predictions,average=None)
    fpr[ii]=fp[ii]/(fp[ii]+tn[ii])
    fnr[ii]=fn[ii]/(fn[ii]+tp[ii])
    CH[ii]=(tp[ii]+fp[ii])*(tp[ii]+fn[ii])/m
    gilbert[ii]=(tp[ii]-CH[ii])/(tp[ii]+fp[ii]+fn[ii]-CH[ii])
    MSE[ii]=mean_squared_error(y_test,predictions)
    tnr[ii]=tn[ii]/(fp[ii]+tn[ii])
    TSS[ii]=tp[ii]/(tp[ii]+fn[ii])-fp[ii]/(fp[ii]+tn[ii])

print 'Mean and Std deviation of Heidke scores (Graham and Leka, 2008)=',np.mean(skill),np.std(skill)
print 'Mean and Std deviation of Heidke scores (Balch 2008)=',np.mean(skill2),np.std(skill2)
print 'Mean and Std deviation of Gilbert scores=',np.mean(gilbert),np.std(gilbert)
print 'Mean and Std deviation of True Skill Scores=',np.mean(TSS),np.std(TSS)
print 'Mean and Std deviation of accuracies=  ',np.mean(acc),np.std(acc)
print 'Mean and Std deviation of precisions (negative)=  ',np.mean(prec[:,0]),np.std(prec[:,0])
print 'Mean and Std deviation of precisions (positive)=  ',np.mean(prec[:,1]),np.std(prec[:,1])
print 'Mean and Std deviation of f1-score (negative)=    ',np.mean(f1[:,0]),np.std(f1[:,0])
print 'Mean and Std deviation of f1-score (positive)=    ',np.mean(f1[:,1]),np.std(f1[:,1])
print 'Mean and Std deviation of recalls/sensitivity (negative)=     ',np.mean(rec[:,0]),np.std(rec[:,0])
print 'Mean and Std deviation of recalls/sensitivity (positive)=     ',np.mean(rec[:,1]),np.std(rec[:,1])
print 'Mean and Std deviation of false positive rate=',np.mean(fpr),np.std(fpr)
print 'Mean and Std deviation of false negative rate=',np.mean(fnr),np.std(fnr)
print 'Mean and Std deviation of specificity=',np.mean(tnr),np.std(tnr)
print 'Mean and Std deviation of MSE=',np.mean(MSE),np.std(MSE)
print 'Mean and Std deviation of number of positive examples in training set=',np.mean(numtraip),np.std(numtraip)
print 'Mean and Std deviation of number of negative examples in training set=',np.mean(numtrain),np.std(numtrain)
print 'ratio=',np.mean(numtrain)/np.mean(numtraip)
print 'Mean and Std deviation of number of positive examples in testing set=',np.mean(numtestp),np.std(numtestp)
print 'Mean and Std deviation of number of negative examples in testing set=',np.mean(numtestn),np.std(numtestn)
print 'ratio=',np.mean(numtestn)/np.mean(numtestp)

end = time.clock()
print 'running time=',end-start 
