'''
Program to forecast solare flares using a Support Vector Machine algorithm
This code uses the scikit-learn Python module
Different catalogs of features of flaring solar active regions and non flaring ones
are used depending on the definition of a flare.
The active-region features are taken from the HMI SHARP parameters (Bobra et al., 2014)
'''

import time, numpy as np, sys, getopt
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score, accuracy_score,f1_score,mean_squared_error
from sklearn.preprocessing import LabelBinarizer, scale
from sklearn.decomposition import PCA 
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, SelectFpr, SelectKBest
from sklearn.utils import shuffle

def main():

   start = time.clock()

   #READ COMMAND-LINE ARGUMENTS
   #-----------------------------------------------------------------------------------------------
   
   try:
       opts, args = getopt.getopt(sys.argv[1:],"hc:",["help","choice="]) #h is for help, c is for choice
   except getopt.GetoptError as err:
       print str(err)
       print 'SVM_flare.py -c <choice>'
       sys.exit(2)

   if len(opts) >= 1:     
          for opt, arg in opts:
              if opt in ("-h", "--help"):
                  print 'SVM_flare.py -c <choice>'
                  sys.exit()
              elif opt in ("-c", "--choice"):
                  print arg
                  choice = int(arg)
              else:
                  assert False, "unhandled option"
                  sys.exit(2)
   else:
          print 'wrong or missing argument:'
          print 'SVM_flare.py -c <choice>'
          sys.exit(2)   
   
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
   
   if (choice == 1):  listAccept =    [0,3,4,5,11,12,13,15,16,19,20,23,24] #Fisher ranking (25 features)
   elif (choice == 2):listAccept =    [5,20,26,13,11,24,12,0,15,19,3,16,23] #Fisher ranking (27 features)
   elif (choice == 3):listAccept =    [16,25,24,26,2,11,12,23,13,10,18,17,3] #mRMRq
   elif (choice == 4):listAccept =    [16,7,25,24,9,21,10,6,22,12,26,18,2] #mRMRd
   elif (choice == 5):listAccept =    [16,25,24,26,11,2,23,12,13,10,18,17,3] #mRMRnq
   elif (choice == 6):listAccept =    [16,7,25,24,9,21,10,6,22,12,26,2,18] #mRMRnd
   elif (choice == 7):listAccept =    [24,16,25,26,12,13,11,7,23,18,10,17,3] #mRMRqSU
   elif (choice == 8):listAccept =    [24,6,25,7,16,22,26,21,10,12,9,2,8] #mRMRdSU
   elif (choice == 9):listAccept =    [24,16,25,26,12,13,11,10,23,18,17,7,3] #mRMRnqSU
   elif (choice == 10):listAccept =   [24,25,16,6,7,26,22,10,12,21,9,13,11] #mRMRndSU
   elif (choice == 0):listAccept = range(27) #all features are kept
   elif (choice == 11):listAccept =   [6,9,7,8,19,1,4,21,22,0,14,15,20] #mRMRq LEAST WELL RANKED
   elif (choice == 12):listAccept =   [5,20,0,4,19,23,15,13,1,14,17,3,11] #mRMRd LEAST WELL RANKED
   elif (choice == 13):listAccept =   [21,22,7,17,10,2,18,6,9,1,25,8,14] #Fisher 27 LEAST WELL RANKED
   elif (choice == 14):listAccept =   [5,20,26,13,11] #only the 5 best features from Fisher 27
   elif (choice == 15):listAccept =   [21,22,7,17,10] #only the 5 worst features from Fisher 27
   elif (choice == 16):listAccept =   [5,20,26,13] #only the 4 best features from Fisher 27
   elif (choice == 17):listAccept =   [21,22,7,17] #only the 4 worst features from Fisher 27
   elif (choice == 18):listAccept =   [5,20,26] #only the 3 best features from Fisher 27
   elif (choice == 19):listAccept =   [21,22,7] #only the 3 worst features from Fisher 27
   elif (choice == 20):listAccept =   [5,20] #only the 2 best features from Fisher 27
   elif (choice == 21):listAccept =   [21,22] #only the 2 worst features from Fisher 27
   elif (choice == 22):listAccept =   [16,25,24,26,2] #mRMRq best 5 features
   elif (choice == 23):listAccept =   [6,9,7,8,19] #mRMRq worst 5 features
   elif (choice == 24):listAccept =   [16,25,24,26] #mRMRq best 4 features
   elif (choice == 25):listAccept =   [6,9,7,8] #mRMRq worst 4 features
   elif (choice == 26):listAccept =   [16,25,24] #mRMRq best 3 features
   elif (choice == 27):listAccept =   [6,9,7] #mRMRq worst 3 features
   elif (choice == 28):listAccept =   [16,25] #mRMRq best 3 features
   elif (choice == 29):listAccept =   [6,9] #mRMRq worst 3 features
   elif (choice == 30):listAccept =   [16,7,25,24,9] #mRMRd best 5 features
   elif (choice == 31):listAccept =   [5,20,0,4,19] #mRMRd worst 5 features
   elif (choice == 32):listAccept =   [16,7,25,24] #mRMRd best 4 features
   elif (choice == 33):listAccept =   [5,20,0,4] #mRMRd worst 4 features
   elif (choice == 34):listAccept =   [16,7,25] #mRMRd best 3 features
   elif (choice == 35):listAccept =   [5,20,0] #mRMRd worst 3 features
   elif (choice == 36):listAccept =   [24,25,16,6,7] #mRMRndSU 5 best
   elif (choice == 37):listAccept =   [4,19,20,0,5] #mRMRndSU 5 worst
   elif (choice == 38):listAccept =   [24,25,16,6] #mRMRndSU 4 best
   elif (choice == 39):listAccept =   [4,19,20,0] #mRMRndSU 4 worst
   elif (choice == 40):listAccept =   [24,25,16,6,7,26] #mRMRndSU 6 best
   elif (choice == 41):listAccept =   [24,25,16,6,7,26,22] #mRMRndSU 6 best
   elif (choice == 42):listAccept =   [16,25,24,26,2,11] #mRMRq best 6 features
   elif (choice == 43):listAccept =   [16,25,24,26,2,11,12] #mRMRq best 7 features
   elif (choice == 44):listAccept =   [16,7,25,24,9,21] #mRMRd best 6 features
   elif (choice == 45):listAccept =   [16,7,25,24,9,21,10] #mRMRd best 6 features
   elif (choice == 46):listAccept =   [5,20,26,13,11,24] #Fisher ranking (27 features) 6 best features
   elif (choice == 47):listAccept =   [5,20,26,13,11,24,12] #Fisher ranking (27 features) 7 best features
   elif (choice == 48):listAccept =   [5,20,26,13,11,24,12,0] #Fisher ranking (27 features) 8 best features
   elif (choice == 49):listAccept =   [5,20,26,13,11,24,12,0,15] #Fisher ranking (27 features) 9 best features
   elif (choice == 50):listAccept =   [5,20,26,13,11,24,12,0,15,19] #Fisher ranking (27 features) 10 best features
   elif (choice == 51):listAccept =   [16,25,24,26,11,2] #mRMRnq 6 best features
   elif (choice == 52):listAccept =   [16,7,25,24,9,21] #mRMRnd 6 best features
   elif (choice == 53):listAccept =   [24,16,25,26,12,13] #mRMRqSU 6 best features
   elif (choice == 54):listAccept =   [5,20,26,13,11,24,12,0,15,19,3] #Fisher ranking (27 features) 11 best features
   elif (choice == 55):listAccept =   [5,20,26,13,11,24,12,0,15,19,3,16] #Fisher ranking (27 features) 12 best features
   elif (choice == 56):listAccept =   [5,20,26,13,11,24] #Fisher ranking (27 features) 6 best features
   elif (choice == 57):listAccept =   [5,20,26,13,11] #Fisher ranking (27 features) 5 best features
   elif (choice == 58):listAccept =   [5,20,26,13,11,24,12,0,15,19,3,16,23,4] #Fisher ranking best 14
   elif (choice == 59):listAccept =   [5,20,26,13,11,24,12,0,15,19,3,16,23,4,14] #Fisher ranking best 15
   elif (choice == 60):listAccept =   [5,20,26,13,11,24,12,0,15,19,3,16,23,4,14,8] #Fisher ranking best 16
   elif (choice == 61):listAccept =   [5,20,26,13,11,24,12,0,15,19,3,16,23,4,14,8,25] #Fisher ranking best 17
   elif (choice == 62):listAccept =   [5,20,26,13,11,24,12,0,15,19,3,16,23,4,14,8,25,1] #Fisher ranking best 18
   elif (choice == 63):listAccept =   [5,20,26,13,11,24,12,0,15,19,3,16,23,4,14,8,25,1,9] #Fisher ranking best 19
   elif (choice == 64):listAccept =   [5,20,26,13,11,24,12,0,15,19,3,16,23,4,14,8,25,1,9,6] #Fisher ranking best 20
   elif (choice == 65):listAccept =   [5,20,26,13,11,24,12,0,15,19,3,16,23,4,14,8,25,1,9,6,18] #Fisher ranking best 21
   elif (choice == 66):listAccept =   [5,20,26,13,11,24,12,0,15,19,3,16,23,4,14,8,25,1,9,6,18,2] #Fisher ranking best 22
   elif (choice == 67):listAccept =   [5,20,26,13,11,24,12,0,15,19,3,16,23,4,14,8,25,1,9,6,18,2,10] #Fisher ranking best 23
   elif (choice == 68):listAccept =   [5,20,26,13,11,24,12,0,15,19,3,16,23,4,14,8,25,1,9,6,18,2,10,17] #Fisher ranking best 24
   elif (choice == 69):listAccept =   [5,20,26,13,11,24,12,0,15,19,3,16,23,4,14,8,25,1,9,6,18,2,10,17,7] #Fisher ranking best 25
   elif (choice == 70):listAccept =   [5,20,26,13,11,24,12,0,15,19,3,16,23,4,14,8,25,1,9,6,18,2,10,17,7,22] #Fisher ranking best 26
   elif (choice == 71):listAccept =   [16,25,24,26,11,2,23] #mRMRnq 7 best features
   elif (choice == 72):listAccept =   [16,25,24,26,11,2,23,12] #mRMRnq 8 best features
   elif (choice == 73):listAccept =   [16,25,24,26,11] #mRMRnq 5 best features
   elif (choice == 74):listAccept =   [16,25,24,26] #mRMRnq 4 best features
   elif (choice == 75):listAccept =   [16,25,24,26,11,2,23,12,13] #mRMRnq 9 best features
   elif (choice == 76):listAccept =   [16,25,24,26,11,2,23,12,13,10] #mRMRnq 10 best features
   elif (choice == 77):listAccept =   [16,25,24,26,11,2,23,12,13,10,18] #mRMRnq 11 best features
   elif (choice == 78):listAccept =   [16,25,24,26,11,2,23,12,13,10,18,17] #mRMRnq 12 best features
   elif (choice == 79):listAccept =   [16,25,24,26,11,2,23,12,13,10,18,17,3] #mRMRnq 13 best features
   elif (choice == 80):listAccept =   [16,25,24,26,11,2,23,12,13,10,18,17,3,5] #mRMRnq 14 best features
   elif (choice == 81):listAccept =   [16,25,24,26,11,2,23,12,13,10,18,17,3,5,20] #mRMRnq 15 best features
   elif (choice == 82):listAccept =   [16,25,24,26,11,2,23,12,13,10,18,17,3,5,20,15] #mRMRnq 16 best features
   elif (choice == 83):listAccept =   [16,25,24,26,11,2,23,12,13,10,18,17,3,5,20,15,14] #mRMRnq 17 best features
   elif (choice == 84):listAccept =   [16,25,24,26,11,2,23,12,13,10,18,17,3,5,20,15,14,0] #mRMRnq 18 best features
   elif (choice == 85):listAccept =   [16,25,24,26,11,2,23,12,13,10,18,17,3,5,20,15,14,0,22] #mRMRnq 19 best features
   elif (choice == 86):listAccept =   [16,25,24,26,11,2,23,12,13,10,18,17,3,5,20,15,14,0,22,21] #mRMRnq 20 best features
   elif (choice == 87):listAccept =   [16,25,24,26,11,2,23,12,13,10,18,17,3,5,20,15,14,0,22,21,7] #mRMRnq 21 best features
   elif (choice == 88):listAccept =   [16,25,24,26,11,2,23,12,13,10,18,17,3,5,20,15,14,0,22,21,7,8] #mRMRnq 22 best features
   elif (choice == 89):listAccept =   [16,25,24,26,11,2,23,12,13,10,18,17,3,5,20,15,14,0,22,21,7,8,1] #mRMRnq 23 best features
   elif (choice == 90):listAccept =   [16,25,24,26,11,2,23,12,13,10,18,17,3,5,20,15,14,0,22,21,7,8,1,19] #mRMRnq 24 best features
   elif (choice == 91):listAccept =   [16,25,24,26,11,2,23,12,13,10,18,17,3,5,20,15,14,0,22,21,7,8,1,19,4] #mRMRnq 25 best features
   elif (choice == 92):listAccept =   [16,25,24,26,11,2,23,12,13,10,18,17,3,5,20,15,14,0,22,21,7,8,1,19,4,9] #mRMRnq 26 best features
   elif (choice == 93):listAccept =   [16,25,24,26,11,2,23,12,13,10,18,17,3,5,20,15,14,0,22,21,7,8,1,19,4,9,6] #mRMRnq 27 best features
   elif (choice == 94):listAccept =   [5,20,26,13] #Fisher ranking (27 features) 4 best features
   elif (choice == 95):listAccept =   [5,20,26] #Fisher ranking (27 features) 3 best features
   elif (choice == 96):listAccept =   [5,20] #Fisher ranking (27 features) 2 best features
   elif (choice == 97):listAccept =   [16,25,24] #mRMRnq 3 best features
   elif (choice == 98):listAccept =   [16,25] #mRMRnq 2 best features
   else:
       print 'choice %d not supported' % choice
       sys.exit(2)
   
   Xflare   = Xflare[:,listAccept]
   Xnoflare = Xnoflare[:,listAccept]
   
   #FEATURE SCALING
   #--------------------------------------------------------------------
   
   for i in range(Xflare.shape[1]):
       meancat=np.median(np.append(Xflare[:,i],Xnoflare[:,i]))
       sigmacat=np.std(np.append(Xflare[:,i],Xnoflare[:,i]))
       Xflare[:,i] = (Xflare[:,i]-meancat)/sigmacat
       Xnoflare[:,i]= (Xnoflare[:,i]-meancat)/sigmacat
   
   selector = SelectKBest(f_classif,k="all")
   selector.fit(np.append(Xflare,Xnoflare,axis=0),np.append(np.ones(Xflare.shape[0]),np.zeros(Xnoflare.shape[0]),axis=0))
   scores=selector.scores_
   print 'Univariate scores before feature selection=',scores
   
   #SVM ALGORITHM
   #-----------------------------------------------------------------
   
   C = 10 # SVM regularization parameter (4 for univariate score, 10 with entropy feature selection and for best HSS2, 1 otherwise)
   clf = svm.SVC(kernel='rbf',gamma=0.08,C=C,class_weight={1:2.0},cache_size=400,tol=1e-8) #parameters for best HSS2: gamma=0.075, class_weight={1:2.00}
   #clf = svm.SVC(kernel='rbf',gamma=0.08,C=C,class_weight={1:17.0},cache_size=400,tol=1e-8) #parameters for best TSS with entropy ranking scores
   #clf = svm.SVC(kernel='rbf',gamma=0.075,C=C,class_weight={1:15.0},cache_size=400,tol=1e-8) #parameters for best TSS with univariate score
   
   # TRAINING AND TESTING THE SVM
   #-----------------------------------------------------------------
   
   ntest = 1000 #number of tests (1 test= select random negative examples to add to the positive ones and create the training set, then preprocess, then train the SVM, then predict on a cross-validation set
   nexamples=5231 #4991 #number of negative examples in training set
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
       
       y  = np.ones(Xflare.shape[0],dtype=np.int16) #labels have to be integer for the SVC function in case of classification
       y  = np.append(y,-1*np.ones(nexamples,dtype=np.int16))             
       
       X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30) #split the example set into a training and testing set
       clf.fit(X_train, y_train)  #train the SVM on the training set
   
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

if __name__ == "__main__":
    main()
