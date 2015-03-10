'''
Program to forecast solare flares using the C4.5 or CART algorithms to build a decision tree
Uses the scikit-learn Python module
Different catalogs of features of flaring solar active regions and non flaring ones
are used depending on the definition of a flare.
Active-region features are taken from the HMI SHARP parameters (Bobra et al., 2014)
'''

import time, numpy as np
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score, accuracy_score,f1_score,mean_squared_error
from sklearn.preprocessing import LabelBinarizer, scale
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

def main():

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
   
   with open('flare_catalog_24h_times_beff_out.txt','r') as flare_file:  #to add the B effective
       Xtext3=flare_file.readlines()
   flare_file.close()
   
   for i in range(len(Xtext)):
       res=Xtext[i].split()
       res2=Xtext2[i].split()
       res3=Xtext3[i].split()
       Xflare.append([float(res[0]),float(res[1]),float(res[2]),float(res[3]),float(res[4]),float(res[5]),float(res[6]),float(res[7]),float(res[8]),float(res[9]),float(res[10]),float(res[11]),float(res[12]),float(res[13]),float(res[14]),float(res[15]),float(res[16]),float(res[17]),float(res[18]),float(res[19]),float(res[20]),float(res[21]),float(res[22]),float(res[23]),float(res[24]),float(res2[0]),float(res3[0])])
   
   #NO-FLARE FEATURE CATALOG
   #---------------------------------------------------------------------
   
   Xnoflare=[]
   #with open('noflare_catalog_24h.txt','r') as noflare_file:  #for active regions not flaring in a 24h interval prior to a flare 
   with open('noflare_catalog_48h.txt','r') as noflare_file:   #for active regions not flaring in a 48h interval around a flare
       Xtext=noflare_file.readlines()
   noflare_file.close()
   
   #with open('noflare_catalog_24h_times_d0_out.txt','r') as noflare_file:  #to add the fractal dimension
   with open('noflare_catalog_48h_times_d0_out.txt','r') as noflare_file:  #to add the fractal dimension
       Xtext2=noflare_file.readlines()
   noflare_file.close()
   
   #with open('noflare_catalog_24h_times_beff_out.txt','r') as noflare_file:  #to add the B effective
   with open('noflare_catalog_48h_times_beff_out.txt','r') as noflare_file:  #to add the B effective
       Xtext3=noflare_file.readlines()
   noflare_file.close()
   
   for i in range(len(Xtext)):
       res=Xtext[i].split()
       res2=Xtext2[i].split()
       res3=Xtext3[i].split()    
       Xnoflare.append([float(res[0]),float(res[1]),float(res[2]),float(res[3]),float(res[4]),float(res[5]),float(res[6]),float(res[7]),float(res[8]),float(res[9]),float(res[10]),float(res[11]),float(res[12]),float(res[13]),float(res[14]),float(res[15]),float(res[16]),float(res[17]),float(res[18]),float(res[19]),float(res[20]),float(res[21]),float(res[22]),float(res[23]),float(res[24]),float(res2[0]),float(res3[0])])
   
   Xflare = np.array(Xflare,dtype=np.float64)
   Xnoflare = np.array(Xnoflare,dtype=np.float64)
   
   listAccept =    [5,20,26,13,11,24,12,0,15,19,3,16,23]
   
   Xflare   = Xflare[:,listAccept]
   Xnoflare = Xnoflare[:,listAccept]
   
   #FEATURE SCALING
   #--------------------------------------------------------------------
   
   for i in range(Xflare.shape[1]):
       meancat=np.median(np.append(Xflare[:,i],Xnoflare[:,i]))
       sigmacat=np.std(np.append(Xflare[:,i],Xnoflare[:,i]))
       Xflare[:,i] = (Xflare[:,i]-meancat)/sigmacat
       Xnoflare[:,i]= (Xnoflare[:,i]-meancat)/sigmacat 
   
   #SELECT DECISION TREE CLASSIFIER
   #-----------------------------------------------------------------
   
   #clf = tree.DecisionTreeClassifier(criterion="entropy") #uses the information gain to measure the split quality (C4.5 algorithm)
   clf = tree.DecisionTreeClassifier(criterion="gini") #uses the Gini impurity to measure the split quality (CART algorithm)

   # TRAINING AND TESTING THE DECISION TREE
   #-----------------------------------------------------------------
   
   ntest = 100 #number of tests 
   nexamples=5231 #number of negative examples in training set
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
       X3=X2[0:nexamples,:]
       X=np.append(X,X3,axis=0)
       
       y  = np.ones(Xflare.shape[0],dtype=np.int16) #labels have to be integer and should be +1 and -1
       y  = np.append(y,-1*np.ones(nexamples,dtype=np.int16))             
       
       X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30) #split the example set into a training and testing set
       clf.fit(X_train, y_train)  #train the decision tree on the training set
   
       #add unused negative examples to the testing set
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
           predictions=np.append(predictions,o)

       #compute performance metrics
       tp[ii]=sum(x>1.0 for x in predictions+y_test)
       tn[ii]=sum(x<-1.0 for x in predictions+y_test)
       fn[ii]=sum(x>1.0 for x in y_test-predictions)
       fp[ii]=sum(x<-1.0 for x in y_test-predictions)
       rec[ii,:]=recall_score(y_test,predictions,average=None)
       prec[ii,:]=precision_score(y_test,predictions,average=None)
       acc[ii]=(tp[ii]+tn[ii])/(tp[ii]+fn[ii]+tn[ii]+fp[ii])
       skill[ii]=(tp[ii]-fp[ii])/(tp[ii]+fn[ii])
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

   print 'DECISION TREE CLASSIFICATION RESULTS:'
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

if __name__ == "__main__":
    main()
