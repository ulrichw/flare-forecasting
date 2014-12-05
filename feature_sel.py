''' Program to rank the SHARP features used for solar flare forecasting
using feature selection algorithms mostly based on Shannon entropy
called like this:
python2.7 feature_sel.py -c 1 -e 26 -l 100
where:
-c is the choice of feature-selection method you want 
-e is the number of SHARP parameters to rank/select (if you want a subset)
-l is the number of discretization levels (the Shannon entropy is computed
using an histogram method, so the number or bins in the histogram directly impacts
entropy computations)
The more levels you have, the slower is the code. We recommend not to exceed a few hundreds.

This program uses the pyentropy module.
Definitions of the routines and entropy estimation methods are in:
Ince, Petersen, Swan, and Panzeri, 2009, Frontiers in Neuroinformatics, 3, 4
'''

import math, numpy as np, pyentropy, sys, getopt
from   sklearn.feature_selection import SelectKBest, f_classif

# FUNCTION DEFINITIONS
#-----------------------------------------------------------------------------------------------

#must discretize the continuous X variables first
#information gain
def IG(X,Y):
    X=X.reshape(m)
    Y=Y.reshape(m)
    s = pyentropy.DiscreteSystem(X,(1,X.max()+1), Y,(1,Y.max()+1))
    s.calculate_entropies(method='qe', calc=['HX', 'HXY','HiXY','HshXY']) #we use the quadratic extrapolation method for entropy estimation
    return s.Ish() #returns shuffled mutual information

#must discretize the continuous X variables first
#symmetrical uncertainty
def SU(X,Y):
    X=X.reshape(m)
    Y=Y.reshape(m)
    s = pyentropy.DiscreteSystem(X,(1,X.max()+1), Y,(1,Y.max()+1))
    s.calculate_entropies(method='qe', calc=['HX', 'HY', 'HXY','HiXY','HshXY'])  #we use the quadratic extrapolation method for entropy estimation
    return 2.0*s.Ish()/(s.H['HX']+s.H['HY'])  #returns shuffled symmetrical information

#READ COMMAND-LINE ARGUMENTS
#-----------------------------------------------------------------------------------------------

try:
    opts, args = getopt.getopt(sys.argv[1:],"hc:e:l:",["choice=","elements=","levels="]) #h is for help, c is for choice
except getopt.GetoptError:
    print 'wrong or missing argument:'
    print 'feature_sel.py -c <choice> -e <elements> -l <levels>'
    sys.exit(2)

if len(opts) != 0:    
    for opt, arg in opts:
        if opt == '-h':
            print 'feature_sel.py -c <choice> -e <elements> -l <levels>'
            sys.exit()
        elif opt in ("-c", "--choice"):
            choice = int(arg)
        elif opt in ("-e", "--elements"):#number of elements we want in the selected features (e.g. 13)
            nelem = int(arg)
        elif opt in ("-l","--levels"):#number of bins for the discretization (e.g. 100)
            nlev = float(arg)
        else:
            print 'wrong or missing argument:'
            print 'feature_sel.py -c <choice> -e <elements> -l <levels>'
            sys.exit(2)
else:
    print 'wrong or missing argument:'
    print 'feature_sel.py -c <choice> -e <elements> -l <levels>'
    sys.exit(2)
        

# READING FLARE AND NO-FLARE CATALOGS
#-----------------------------------------------------------------------------------------------

#FLARE CATALOG
with open('flare_catalog_24h.txt','r') as flare_file:                 #the 25 SHARP parameters
    Xtext=flare_file.readlines()
flare_file.close()
Xflare=[]

with open('flare_catalog_24h_times_d0_out.txt','r') as flare_file:    #to add the fractal dimension
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

#NO-FLARE CATALOG
Xnoflare=[]
with open('noflare_catalog_48h.txt','r') as noflare_file:                 #the 25 SHARP parameters
    Xtext=noflare_file.readlines()
noflare_file.close()

with open('noflare_catalog_48h_times_d0_out.txt','r') as noflare_file:    #to add the fractal dimension
    Xtext2=noflare_file.readlines()
noflare_file.close()

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

#names of SHARP features
names=['USFLUX','MEANGBT','MEANJZH','MEANPOT','SHRGT45','TOTUSJH','MEANGBH','MEANALP','MEANGAM','MEANGBZ','MEANJZD','TOTUSJZ','SAVNCPP','TOTPOT','MEANSHR','AREA_ACR','R_VALUE','TOTFX','TOTFY','TOTFZ','TOTBSQ','EPSX','EPSY','EPSZ','ABSNJZH','fractal','B effective']
names=np.array(names)

#PRE-PROCESSING
#-----------------------------------------------------------------------------------------------

#feature scaling
for i in range(Xflare.shape[1]):
    meancat=np.min(np.append(Xflare[:,i],Xnoflare[:,i]))
    sigmacat=np.max(np.append(Xflare[:,i],Xnoflare[:,i]))-np.min(np.append(Xflare[:,i],Xnoflare[:,i]))
    Xflare[:,i]  = (Xflare[:,i]-meancat)  /sigmacat*nlev
    Xnoflare[:,i]= (Xnoflare[:,i]-meancat)/sigmacat*nlev

X=np.append(Xflare,Xnoflare,axis=0)
Y=np.append(np.ones(Xflare.shape[0]),np.zeros(Xnoflare.shape[0]),axis=0)
N=Xflare.shape[1] #number of features
m=X.shape[0]      #number of examples

X=np.around(X)    #rounding the numbers
Y=np.around(Y)
X=X.astype(np.int32,copy=False) #converts from float to integer
Y=Y.astype(np.int32,copy=False)


#univariate feature selection
#-----------------------------------------------------------------------------------------------

if choice == 0:
  selector = SelectKBest(f_classif,k="all")
  selector.fit(X,Y)
  scores=selector.scores_
    
  print 'Univariate Fisher score:'
  print scores


#mRMR algorithm with quotient
#from Peng (2005)
#-------------------------------------------------------------

if choice == 1:
    
  output=[] #will contain the indices of the selected features
  
  #compute the relevances of all the features
  relevance=[]
  for i in range(N):
      SU0=IG(X[:,i],Y)
      relevance.append(SU0) #relevance of feature i
  
  #find the highest relevance
  index0=[]
  index0.append(relevance.index(max(relevance))) #returns the 1st index of the maximum relevance
  
  #compute the redundancies between the first selected feature and all of the remaining ones
  while len(index0) < nelem :
      redundancy=[]
      for i in range(N):
          if i not in index0:
              red=0.0
              nred=0.0
              for j in range(N):
                  if j in index0:
                      red += IG(X[:,j],X[:,i])
                      nred +=1.0
              redundancy.append(red/nred)
          else:
              redundancy.append(1e10)
  
      #compute relevance/redundancy
      objective=[]
      for i in range(N):
          objective.append(relevance[i]/redundancy[i]) #we use the quotient version of the mRMR criterion
          
      #locate minimum of relevance/redundancy
      
  
      #add to the list of selected features
      index0.append(objective.index(max(objective)))
  
  print 'mRMR quotient'
  print 'relevance=',relevance
  print index0
  

#mRMR algorithm with difference
#from Peng (2005)
#-------------------------------------------------------------

if choice == 2:

  output=[] #will contain the indices of the selected features
  
  #compute the relevances of all the features
  relevance=[]
  for i in range(N):
      SU0=IG(X[:,i],Y)
      relevance.append(SU0) #relevance of feature i
  
  #find the highest relevance
  index0=[]
  index0.append(relevance.index(max(relevance))) #returns the 1st index of the maximum relevance
  
  #compute the redundancies between the first selected feature and all of the remaining ones
  while len(index0) < nelem :
      redundancy=[]
      for i in range(N):
          if i not in index0:
              red=0.0
              nred=0.0
              for j in range(N):
                  if j in index0:
                      red += IG(X[:,j],X[:,i])
                      nred +=1.0
              redundancy.append(red/nred)
          else:
              redundancy.append(1e10)
  
      #compute relevance/redundancy
      objective=[]
      for i in range(N):
          objective.append(relevance[i]-redundancy[i]) #we use the difference version of the mRMR criterion
          
      #locate minimum of relevance-redundancy
      
  
      #add to the list of selected features
      index0.append(objective.index(max(objective)))
  
  print 'mRMR difference'
  print index0
  


#mRMR algorithm with quotient
#with non-dominated condition
#-------------------------------------------------------------

if choice == 3:
    
  output=[] #will contain the indices of the selected features
  
  #compute the relevances of all the features
  relevance=[]
  for i in range(N):
      SU0=IG(X[:,i],Y)
      relevance.append(SU0) #relevance of feature i
  
  #find the highest relevance
  index0=[]
  index0.append(relevance.index(max(relevance))) #returns the 1st index of the maximum relevance
  
  #compute the redundancies between the first selected feature and all of the remaining ones
  while len(index0) < nelem :
      redundancy=[]
      for i in range(N):
          if i not in index0:
              red=0.0
              nred=0.0
              for j in range(N):
                  if j in index0:
                      red += IG(X[:,j],X[:,i])
                      nred +=1.0
              redundancy.append(red/nred)
          else:
              redundancy.append(1e10)
  
      #compute relevance/redundancy
      objective=[]
      for i in range(N):
          objective.append(relevance[i]/redundancy[i]) #we use the quotient version of the mRMR criterion
          
      #locate non-dominated features
      nondominated=[]
      for i in range(N):
          dominated=0
          if (i not in index0):
              for j in range(N):
                  if (j not in index0) and (relevance[j] > relevance[i]) and (redundancy[j] < redundancy[i]):   #i is dominated by j
                      dominated=1
              if dominated == 0:
                  nondominated.append(i) #add the i feature to the non-dominated list
  
      #from the list of non-dominated features select the one with the highest relevance/redundancy ratio
      index0.append(objective.index(max(objective)))
  
  print 'mRMR with quotient and non-dominated condition:'
  print index0



#mRMR algorithm with difference
#with non-dominated condition
#-------------------------------------------------------------

if choice == 4:
    
  output=[] #will contain the indices of the selected features
  
  #compute the relevances of all the features
  relevance=[]
  for i in range(N):
      SU0=IG(X[:,i],Y)
      relevance.append(SU0) #relevance of feature i
  
  #find the highest relevance
  index0=[]
  index0.append(relevance.index(max(relevance))) #returns the 1st index of the maximum relevance
  
  #compute the redundancies between the first selected feature and all of the remaining ones
  while len(index0) < nelem :
      redundancy=[]
      for i in range(N):
          if i not in index0:
              red=0.0
              nred=0.0
              for j in range(N):
                  if j in index0:
                      red += IG(X[:,j],X[:,i])
                      nred +=1.0
              redundancy.append(red/nred)
          else:
              redundancy.append(1e10)
  
      #compute relevance/redundancy
      objective=[]
      for i in range(N):
          objective.append(relevance[i]-redundancy[i]) #we use the difference version of the mRMR criterion
          
      #locate non-dominated features
      nondominated=[]
      for i in range(N):
          dominated=0
          if (i not in index0):
              for j in range(N):
                  if (j not in index0) and (relevance[j] > relevance[i]) and (redundancy[j] < redundancy[i]):   #i is dominated by j
                      dominated=1
              if dominated == 0:
                  nondominated.append(i) #add the i feature to the non-dominated list
  
      #from the list of non-dominated features select the one with the highest relevance/redundancy ratio
      index0.append(objective.index(max(objective)))
  
  print 'mRMR with difference and with non-dominated condition:'
  print index0

#Fast Correlation-Based Filter (yu and Liu, 2003)
#[NB FCBF IS USELESS WITH SHARP PARAMETERS, SO RETURN SYMMETRICAL UNCERTAINTY INSTEAD
#------------------------------------------------------------------------------------

if choice == 5:

  #compute the symmetrical uncertainties of all the features with the classes
  symmetricalun=[]
  for i in range(N):
      SU0=SU(X[:,i],Y)
      symmetricalun.append(SU0) #symmetrical uncertainty of feature i
  #sort symmetrical uncertainties in descending order
  SUsorted=sorted(range(len(symmetricalun)),key=lambda k: symmetricalun[k],reverse=True) #indices of the symmetrical uncertainties
  symmetricalun=np.array(symmetricalun)
  SUsorted=np.array(SUsorted)
  
  print 'Symmetrical uncertainty:'
  print SUsorted

'''
  selectedfeature=[]
  selectedfeature.append(SUsorted[0]) #1st feature selected
  counter=0
  Fp=0
  while counter < nelem:
      if(Fp+1 < N):
          Fq=Fp+1
          while Fq <nelem:
              Fqp=Fq
              print Fq,Fp
              print SUsorted
              wait = input("PRESS A KEY + ENTER")
  
              if( SU(X[:,SUsorted[Fq]],X[:,SUsorted[Fp]]) >= symmetricalun[SUsorted[Fq]] ):
                  print SU(X[:,SUsorted[Fq]],X[:,SUsorted[Fp]]),symmetricalun[SUsorted[Fq]]
                  SUsorted=np.delete(SUsorted,Fq) #remove by index, not by value
                  Fq=Fqp
              else:
                  Fq=Fq+1
      Fp=Fp+1            
      counter += 1 #counter is the number of features selected
  
  print 'FCBF:'
  print SUsorted
'''

#mRMR algorithm with quotient using SU
#adapted from Peng (2005)
#-------------------------------------------------------------

if choice == 6:
    
  output=[] #will contain the indices of the selected features
  
  #compute the relevances of all the features
  relevance=[]
  for i in range(N):
      SU0=SU(X[:,i],Y)
      relevance.append(SU0) #relevance of feature i
  
  #find the highest relevance
  index0=[]
  index0.append(relevance.index(max(relevance))) #returns the 1st index of the maximum relevance
  
  #compute the redundancies between the first selected feature and all of the remaining ones
  while len(index0) < nelem :
      redundancy=[]
      for i in range(N):
          if i not in index0:
              red=0.0
              nred=0.0
              for j in range(N):
                  if j in index0:
                      red += SU(X[:,j],X[:,i])
                      nred +=1.0
              redundancy.append(red/nred)
          else:
              redundancy.append(1e10)
  
      #compute relevance/redundancy
      objective=[]
      for i in range(N):
          objective.append(relevance[i]/redundancy[i]) #we use the quotient version of the mRMR criterion
          
      #locate minimum of relevance/redundancy
      
  
      #add to the list of selected features
      index0.append(objective.index(max(objective)))
  
  print 'mRMR quotient with Symmetrical uncertainty'
  print 'relevance=',relevance
  print index0
  

#mRMR algorithm with difference using SU
#adapted from Peng (2005)
#-------------------------------------------------------------

if choice == 7:
    
  output=[] #will contain the indices of the selected features
  
  #compute the relevances of all the features
  relevance=[]
  for i in range(N):
      SU0=SU(X[:,i],Y)
      relevance.append(SU0) #relevance of feature i
  
  #find the highest relevance
  index0=[]
  index0.append(relevance.index(max(relevance))) #returns the 1st index of the maximum relevance
  
  #compute the redundancies between the first selected feature and all of the remaining ones
  while len(index0) < nelem :
      redundancy=[]
      for i in range(N):
          if i not in index0:
              red=0.0
              nred=0.0
              for j in range(N):
                  if j in index0:
                      red += SU(X[:,j],X[:,i])
                      nred +=1.0
              redundancy.append(red/nred)
          else:
              redundancy.append(1e10)
  
      #compute relevance/redundancy
      objective=[]
      for i in range(N):
          objective.append(relevance[i]-redundancy[i]) #we use the quotient version of the mRMR criterion
          
      #locate minimum of relevance/redundancy
      
  
      #add to the list of selected features
      index0.append(objective.index(max(objective)))
  
  print 'mRMR difference with Symmetrical uncertainty'
  print index0
  

#mRMR algorithm with quotient and SU
#with non-dominated condition
#-------------------------------------------------------------

if choice == 8:
    
  output=[] #will contain the indices of the selected features
  
  #compute the relevances of all the features
  relevance=[]
  for i in range(N):
      SU0=SU(X[:,i],Y)
      relevance.append(SU0) #relevance of feature i
  
  #find the highest relevance
  index0=[]
  index0.append(relevance.index(max(relevance))) #returns the 1st index of the maximum relevance
  
  #compute the redundancies between the first selected feature and all of the remaining ones
  while len(index0) < nelem :
      redundancy=[]
      for i in range(N):
          if i not in index0:
              red=0.0
              nred=0.0
              for j in range(N):
                  if j in index0:
                      red += SU(X[:,j],X[:,i])
                      nred +=1.0
              redundancy.append(red/nred)
          else:
              redundancy.append(1e10)
  
      #compute relevance/redundancy
      objective=[]
      for i in range(N):
          objective.append(relevance[i]/redundancy[i]) #we use the quotient version of the mRMR criterion
          
      #locate non-dominated features
      nondominated=[]
      for i in range(N):
          dominated=0
          if (i not in index0):
              for j in range(N):
                  if (j not in index0) and (relevance[j] > relevance[i]) and (redundancy[j] < redundancy[i]):   #i is dominated by j
                      dominated=1
              if dominated == 0:
                  nondominated.append(i) #add the i feature to the non-dominated list
  
      #from the list of non-dominated features select the one with the highest relevance/redundancy ratio
      index0.append(objective.index(max(objective)))
  
  print 'mRMR with quotient and non-dominated condition and symmetrical uncertainty:'
  print index0
 


#mRMR algorithm with difference and SU
#with non-dominated condition
#-------------------------------------------------------------

if choice == 9:
    
  output=[] #will contain the indices of the selected features
  
  #compute the relevances of all the features
  relevance=[]
  for i in range(N):
      SU0=SU(X[:,i],Y)
      relevance.append(SU0) #relevance of feature i
  
  #find the highest relevance
  index0=[]
  index0.append(relevance.index(max(relevance))) #returns the 1st index of the maximum relevance
  
  #compute the redundancies between the first selected feature and all of the remaining ones
  while len(index0) < nelem :
      redundancy=[]
      for i in range(N):
          if i not in index0:
              red=0.0
              nred=0.0
              for j in range(N):
                  if j in index0:
                      red += SU(X[:,j],X[:,i])
                      nred +=1.0
              redundancy.append(red/nred)
          else:
              redundancy.append(1e10)
  
      #compute relevance/redundancy
      objective=[]
      for i in range(N):
          objective.append(relevance[i]-redundancy[i]) #we use the difference version of the mRMR criterion
          
      #locate non-dominated features
      nondominated=[]
      for i in range(N):
          dominated=0
          if (i not in index0):
              for j in range(N):
                  if (j not in index0) and (relevance[j] > relevance[i]) and (redundancy[j] < redundancy[i]):   #i is dominated by j
                      dominated=1
              if dominated == 0:
                  nondominated.append(i) #add the i feature to the non-dominated list
  
      #from the list of non-dominated features select the one with the highest relevance/redundancy ratio
      index0.append(objective.index(max(objective)))
  
  print 'mRMR with difference and with non-dominated condition and symmetrical uncertainty:'
  print index0
 
