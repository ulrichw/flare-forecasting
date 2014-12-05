'''
Implementation of a multilayer feed-forward neural network with on-line backpropagation training rule
and Thikonov regularization (with identity matrix)
bias units are added to each layer (except the output one)
x is the design matrix of dimensions m*n where m is the number of examples
and n is the number of features
y is the target/output matrix of dimensions m*K where K is the number of
classes in the multi-class classification.
This program uses the scikit-learn module
''' 

import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score, accuracy_score, f1_score, mean_squared_error
from sklearn.preprocessing import LabelBinarizer, scale
from sklearn.decomposition import PCA 
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.utils import shuffle

#logistic activation function with steepness=1
def logistic(z):
    return 1./(1.+np.exp(-z))
def dlogistic(z):
    return logistic(z)*(1.-logistic(z))

#hyperbolic tangent (symmetrical logistic function)
def tanh(z):
    return np.tanh(z)
def dtanh(z):
    return 1.0 - np.tanh(z)**2

#Elliot activation function with steepness=1
def elliot(z):
    return 0.5*z/(1.+abs(z))+0.5
def delliot(z):
    return 1./(2.*(1.+abs(z))*(1.+abs(z)))

#Elliot symmetric activation function with steepness=1
def elliots(z):
    return z/(1.+abs(z))
def delliots(z):
    z=1.+abs(z)
    return 1./(z*z)

#modified tanh
def tanhm(z):
    return 1.7159*np.tanh(2./3.*z)
def dtanhm(z):
    return 1.7159*2./3.*(1.0 - np.tanh(2./3.*z)**2)

class Network:

    def __init__(self, layers, activation='logistic',learning_rate=0.2,regularization=0.0,init_epsilon=0.25,bias=1.0):
        '''layers is an array describing the number of layers (incl. input and output layers) and the number of unit per layer
        e.g. layer=[10,4,2] means 3 layers: 1 input, 1 hidden, and 1 output, with the input layer having 10 units,
        the hidden layer having 4, and the output having 2 (binary classification)
        the weights are randomly initialiazed in the range [-init_epsilon,+init_epsilon]'''
        if activation == 'logistic':
            self.activation = logistic
            self.activation_der=dlogistic
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_der=dtanh
        elif activation == 'elliot':
            self.activation = elliot
            self.activation_der=delliot
        elif activation == 'elliots':
            self.activation = elliots
            self.activation_der=delliots
        elif activation == 'tanhm':
            self.activation = tanhm
            self.activation_der=dtanhm
             
            
        self.learning_rate=learning_rate
        self.regularization=float(regularization)
        self.layers=layers
        self.theta=[]           #the weights theta for each unit
        self.a=[]               #activation values a for each unit
        self.L=len(self.layers) #number of layers
        self.bias=float(bias)
        #initializes the weights
        for l in range(0,self.L-1):
            self.theta.append((2.*np.random.random((self.layers[l+1],self.layers[l]+1))-1.0)*init_epsilon) #+1 due to bias units for input and hidden layers
        
    def backpropagation(self,x,y,m):
        #computes the gradient of the cost function
        #we loop over all of the examples in the training set

        self.a=[x]                                                                       #feature vector for training example i
        self.a[0]=np.append(self.a[0],self.bias)                                         #add the bias unit
        self.forwardpropagation(x)
        delta=[[self.a[-1]-y]]                                                           #error at each node of output layer (NB: delta is a list of Numpy ndarrays)
        delta[-1]=np.append(delta[-1],0.0)
        for l in range(self.L-2,0,-1):
            temp=np.dot(self.theta[l-1],self.a[l-1])
            delta.append(np.dot(self.theta[l].T,(delta[-1])[:len(delta[-1])-1])*np.append(self.activation_der(temp),0.0)) #add delta=0 for a bias unit 

        delta=delta[::-1]    #reverse delta
        
        #simple gradient descent algorithm
        for l in range(self.L-2):
            temp=np.array(self.theta[l])
            temp[:,-1]=0. #to make sure we will not regularize the bias
            self.theta[l] -= self.learning_rate/m*((np.mat((delta[l])[:len(delta[l])-1]).T*np.mat(self.a[l]))+self.regularization*temp)
        temp=np.array(self.theta[self.L-2])
        temp[:,-1]=0.        
        self.theta[self.L-2] -= self.learning_rate/m*((np.mat((delta[self.L-2])[:len(delta[self.L-2])-1]).T*np.mat(self.a[self.L-2]))+self.regularization*temp)


    def forwardpropagation(self,x):
        ''' a is the matrix of activation values
        self.theta is the vector of weights/parameters
        L is the number of layers (incl. input and output layers)
        i is the index of the selected example'''
        self.a=[x]                                                                       #feature vector for training example i
        self.a[0]=np.append(self.a[0],self.bias)                                         #add the bias unit
        for l in range(1,self.L-1):
            self.a.append(self.activation(np.dot(self.theta[l-1],self.a[l-1])))          #compute linear function and apply activation function
            self.a[l]=np.append(self.a[l],self.bias)                                     #add the bias unit
        self.a.append(self.activation(np.dot(self.theta[self.L-2],self.a[self.L-2])))    #no bias unit in the output layer

    def fit(self,x,y,niteration):
        '''niteration is the number of iterations'''
        x=np.array(x)
        m=x.shape[0] #number of training examples
        for j in range(niteration):
            i = np.random.randint(m)
            self.backpropagation(x[i],y[i],float(m))
 
        
    def predict(self,x):
        a=np.array(x,dtype=np.float64)
        a=np.append(a,self.bias)                          #add the bias unit
        for l in range(1,self.L-1):
            a=self.activation(np.dot(self.theta[l-1],a))
            a=np.append(a,self.bias)                      #add the bias unit
        a=self.activation(np.dot(self.theta[self.L-2],a)) #no bias unit in the output layer
        return a


if __name__ == '__main__':

    start = time.clock()

    #TO TEST THE WORKING OF THE NETWORK:
    #XOR FUNCTION
    '''
    nn=Network([2,2,1],learning_rate=.2,regularization=0.0,init_epsilon=0.25,bias=1,activation='logistic')
    
    x = np.array([[0, 0],
    [0, 1],
    [1, 0],
    [1, 1]],dtype=np.float64)
    y = np.array([0, 1, 1, 0],dtype=np.float64)
    
    nn.fit(x,y,10000)
    
    for i in [[0, 0], [0, 1], [1, 0], [1,1]]:
    print(i,nn.predict(i))'''

    #DIGIT EXAMPLS FROM SCIKIT-LEARN
    '''digits = load_digits()
    X = digits.data
    y = digits.target
    #feature scaling
    X -= X.min()     # normalize the values to bring them into the range 0-1
    X /= X.max()

    #the learning rate depends on: batch or on-line training (can use larger rate when on-line),
    #and the network topology
    nn = Network([64,50,10],learning_rate=6.,regularization=0.1,init_epsilon=0.25,bias=1,activation='tanhm')
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)
    nn.fit(X_train,labels_train,60000) #200000
    predictions = []
    for i in range(X_test.shape[0]):
        o = nn.predict(X_test[i])
        predictions.append(np.argmax(o))
    #print confusion_matrix(y_test,predictions)
    print classification_report(y_test,predictions)
        
    end = time.clock()
    print 'running time=',end-start '''

    #IRIS EXAMPLE FROM SCIKIT-LEARN
    '''iris = load_iris()
    X = iris.data
    y = iris.target
    #pre-processing: feature scaling
    X = preprocessing.scale(X)
    nn = Network([4,8,6,3],learning_rate=1.5,regularization=0.0,init_epsilon=0.25,bias=1,activation='elliots')
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    labels_train = LabelBinarizer().fit_transform(y_train)
    nn.fit(X_train,labels_train,30000)
    predictions = []
    errors = []
    m=float(X_test.shape[0])
    for i in range(X_test.shape[0]):
        o = nn.predict(X_test[i])
        predictions.append(np.argmax(o))
        errors.append( (1./2./m)*(y_test[i]-np.argmax(o))**2)
    print 'Cost=',sum(errors)
    print classification_report(y_test,predictions)
        
    end = time.clock()
    print 'running time=',end-start 

    reg=[0,0.001,0.005,0.01,0.02,0.03,0.05]
    cost=[]'''

    #TO TRAIN AND TEST THE NETWORK ON FLARE AND NO-FLARE SOLAR CATALOGS
    #FLARE CATALOG
    with open('flare_catalog_24h.txt','r') as flare_file:
        Xtext=flare_file.readlines()
    flare_file.close()
    Xflare=[]
    for i in range(len(Xtext)):
        res=Xtext[i].split()
        Xflare.append([float(res[0]),float(res[1]),float(res[2]),float(res[3]),float(res[4]),float(res[5]),float(res[6]),float(res[7]),float(res[8]),float(res[9]),float(res[10]),float(res[11]),float(res[12]),float(res[13]),float(res[14]),float(res[15]),float(res[16])])

    #NO-FLARE CATALOG
    Xnoflare=[]
    with open('noflare_catalog_24h_3.txt','r') as noflare_file:
        Xtext=noflare_file.readlines()
    noflare_file.close()

    for i in range(len(Xtext)):
        res=Xtext[i].split()
        Xnoflare.append([float(res[0]),float(res[1]),float(res[2]),float(res[3]),float(res[4]),float(res[5]),float(res[6]),float(res[7]),float(res[8]),float(res[9]),float(res[10]),float(res[11]),float(res[12]),float(res[13]),float(res[14]),float(res[15]),float(res[16])])

    Xflare = np.array(Xflare,dtype=np.float64)
    Xnoflare = np.array(Xnoflare,dtype=np.float64)
    #Xflare = np.delete(Xflare,[1,2,6,7,9,10],1) #remove features with a low univariate score
    #Xnoflare = np.delete(Xnoflare,[1,2,6,7,9,10],1) #remove features with a low univariate score
    Xflare = np.delete(Xflare,[1],1) #remove features with a low univariate score
    Xnoflare = np.delete(Xnoflare,[1],1) #remove features with a low univariate score

    #feature scaling
    for i in range(Xflare.shape[1]):
        meancat=np.mean(np.append(Xflare[:,i],Xnoflare[:,i]))
        sigmacat=np.std(np.append(Xflare[:,i],Xnoflare[:,i]))
        Xflare[:,i] = (Xflare[:,i]-meancat)/sigmacat
        Xnoflare[:,i]= (Xnoflare[:,i]-meancat)/sigmacat
    
    ninput=Xflare.shape[1]

    ntest=10 #number of tests
    nexamples=9990 #number of negative examples
    rec=np.zeros(ntest)
    rec   = np.zeros(ntest) #recall (sensitivity)
    prec  = np.zeros(ntest) #precision
    acc   = np.zeros(ntest) #accuracy
    skill = np.zeros(ntest) #Heidke skill score from Barnes and Leka (2008)
    skill2= np.zeros(ntest) #Heidke skill score from Balch (2008) and the Space Weather Prediction Center
    f1    = np.zeros(ntest) #F1-score
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
    numtraip=np.zeros(ntest)#number of positive examples in the training set
    numtrain=np.zeros(ntest)#number of negatives examples in the training set
    numtestp=np.zeros(ntest)#number of positive examples in the testing set
    numtestn=np.zeros(ntest)#number of negative examples in the testing set
 
    lr=[13.5]
    for kk in range(len(lr)):

        for ii in range(ntest):

            #create training set
            X=np.array(Xflare)            #positive examples
            X2=np.array(Xnoflare)
            np.random.shuffle(X2)         #shuffle catalog of no-flares
            X3=X2[0:nexamples,:]          #sub-sampling to avoif imbalanced training set
            X=np.append(X,X3,axis=0)
            
            y  = np.ones(Xflare.shape[0])
            y  = np.append(y,-1.*np.ones(nexamples))             

            #nn = Network([ninput,20,10,5,1],learning_rate=2.0,regularization=0.0,init_epsilon=0.25,bias=1,activation='tanh')
            #nn = Network([ninput,27,27,1],learning_rate=lr[kk],regularization=0.0,init_epsilon=0.15,bias=1,activation='tanh')
            nn = Network([ninput,ninput*15,ninput*15,1],learning_rate=lr[kk],regularization=0.0,init_epsilon=0.10,bias=1,activation='tanh')
            X_train, X_test, y_train, y_test = train_test_split(X, y)

            #create testing set
            X_test=np.append(X_test,X2[nexamples:,],axis=0)
            y_test=np.append(y_test,-1.*np.ones(X2.shape[0]-nexamples))
            
            numtraip[ii]=sum([yo>0 for yo in y_train])
            numtrain[ii]=sum([yo<0 for yo in y_train])
            numtestp[ii]=sum([yo>0 for yo in y_test])
            numtestn[ii]=sum([yo<0 for yo in y_test])

            #labels_train = LabelBinarizer(neg_label=-1).fit_transform(y_train)
            #nn.fit(X_train,labels_train,20000) #20000
            nn.fit(X_train,y_train,2000) #20000
            predictions = []
            predictions = np.array(predictions,dtype=np.float64)

            m=float(X_test.shape[0])

            for i in range(X_test.shape[0]):
                o = nn.predict(X_test[i])
                if o <= 0.0:
                    predictions=np.append(predictions,-1.0)
                elif o > 0.0:
                    predictions=np.append(predictions,1.0)

            #print classification_report(y_test,predictions)
            tp[ii]=sum(x>1.0 for x in predictions+y_test)
            tn[ii]=sum(x<-1.0 for x in predictions+y_test)
            fn[ii]=sum(x>1.0 for x in y_test-predictions)
            fp[ii]=sum(x<-1.0 for x in y_test-predictions)
            rec[ii]=tp[ii]/(tp[ii]+fn[ii]) #recall_score(y_test,predictions)
            prec[ii]=tp[ii]/(tp[ii]+fp[ii])#precision_score(y_test,predictions)
            acc[ii]=(tp[ii]+tn[ii])/(tp[ii]+fn[ii]+tn[ii]+fp[ii])#accuracy_score(y_test,predictions)
            skill[ii]=(tp[ii]-fp[ii])/(tp[ii]+fn[ii])#rec[ii]*(2.0-1./prec[ii])
            E[ii]=((tp[ii]+fp[ii])*(tp[ii]+fn[ii])+(fp[ii]+tn[ii])*(fn[ii]+tn[ii]))/m
            skill2[ii]=(tp[ii]+tn[ii]-E[ii])/(tp[ii]+fp[ii]+fn[ii]+tn[ii]-E[ii])
            f1[ii]=f1_score(y_test,predictions)
            fpr[ii]=fp[ii]/(fp[ii]+tn[ii])
            fnr[ii]=fn[ii]/(fn[ii]+tp[ii])
            CH[ii]=(tp[ii]+fp[ii])*(tp[ii]+fn[ii])/m
            gilbert[ii]=(tp[ii]-CH[ii])/(tp[ii]+fp[ii]+fn[ii]-CH[ii])
            MSE[ii]=mean_squared_error(y_test,predictions)
            tnr[ii]=tn[ii]/(fp[ii]+tn[ii])
          
        print 'Learning rate=',lr[kk]
        print 'Mean and Std deviation of Heidke scores (Graham and Leka, 2008)=',np.mean(skill),np.std(skill)
        print 'Mean and Std deviation of Heidke scores (Balch 2008)=',np.mean(skill2),np.std(skill2)
        print 'Mean and Std deviation of Gilbert scores=',np.mean(gilbert),np.std(gilbert)
        print 'Mean and Std deviation of accuracies=  ',np.mean(acc),np.std(acc)
        print 'Mean and Std deviation of precisions=  ',np.mean(prec),np.std(prec)
        print 'Mean and Std deviation of f1-score=    ',np.mean(f1),np.std(f1)
        print 'Mean and Std deviation of false positive rate=',np.mean(fpr),np.std(fpr)
        print 'Mean and Std deviation of false negative rate=',np.mean(fnr),np.std(fnr)
        print 'Mean and Std deviation of specificity=',np.mean(tnr),np.std(tnr)
        print 'Mean and Std deviation of MSE=',np.mean(MSE),np.std(MSE)
        print 'Mean and Std deviation of recalls/sensitivity=     ',np.mean(rec),np.std(rec)      

        print 'Mean and Std deviation of number of positive examples in training set=',np.mean(numtraip),np.std(numtraip)
        print 'Mean and Std deviation of number of negative examples in training set=',np.mean(numtrain),np.std(numtrain)
        print 'ratio=',np.mean(numtrain)/np.mean(numtraip)
        print 'Mean and Std deviation of number of positive examples in testing set=',np.mean(numtestp),np.std(numtestp)
        print 'Mean and Std deviation of number of negative examples in testing set=',np.mean(numtestn),np.std(numtestn)
       
        end = time.clock()
        print 'running time=',end-start 
