'''
Program to compute the Pearson correlation coefficient and R squared
between two active-region features or for all possible combinations
of features
'''

import numpy as np
from   scipy import stats
import sys, getopt
import matplotlib.pyplot as plt

def combinations(iterable, r):
    '''function that computes all combinations from
    elements in the list iterable taken r at a time'''
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    comb=[]
    comb.append(tuple(pool[i] for i in indices))
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return comb
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        comb.append(tuple(pool[i] for i in indices))
    return comb

def main():

    #READ COMMAND-LINE ARGUMENTS
    try:
        opts, args = getopt.getopt(sys.argv[1:],"ha:b:",["help","a=","b="]) #h is for help, c is for choice
    except getopt.GetoptError as err:
        print str(err)
        print 'feature_pearson.py -a <feature1> -b <feature2>'
        print 'if you want to compute the Pearson correlation for all possible combinations'
        print 'of features, use a=-1 and b=-1'
        sys.exit(2)

    if len(opts) >= 2:     
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print 'feature_pearson.py -a <feature1> -b <feature2>'
                print 'if you want to compute the Pearson correlation for all possible combinations'
                print 'of features, use a=-1 and b=-1'
                sys.exit()
            elif opt in ("-a", "--a"):
                feature1 = int(arg)
            elif opt in ("-b", "--b"):
                feature2 = int(arg)                  
            else:
                assert False, "unhandled option"
                sys.exit(2)
    else:
        print 'wrong or missing argument:'
        print 'feature_pearson.py -a <feature1> -b <feature2>'
        print 'if you want to compute the Pearson correlation for all possible combinations'
        print 'of features, use a=-1 and b=-1'
        sys.exit(2)   
   


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

    #make sure both features have the same variance
    for i in range(Xflare.shape[1]):
        meancat=np.min(np.append(Xflare[:,i],Xnoflare[:,i]))
        sigmacat=np.max(np.append(Xflare[:,i],Xnoflare[:,i]))-np.min(np.append(Xflare[:,i],Xnoflare[:,i]))
        Xflare[:,i]  = (Xflare[:,i]-meancat)  /sigmacat
        Xnoflare[:,i]= (Xnoflare[:,i]-meancat)/sigmacat
        
    X=np.append(Xflare,Xnoflare,axis=0)
    Y=np.append(np.ones(Xflare.shape[0]),np.zeros(Xnoflare.shape[0]),axis=0)
    N=Xflare.shape[1] #number of features
    m=X.shape[0]      #number of examples

    if (feature1 != -1 and feature2 != -1):

        #compute Pearson correlation coefficient and p-value for testing non-correlation
        #p-value indicates the probability of an uncorrelated system producing datasets that have a correlation at least
        #as extreme as the one computed from these datasets
        (corr,pvalue) = stats.pearsonr(Xflare[:,feature1],Xflare[:,feature2])
        print "Results for features:",names[feature1],"and",names[feature2]
        print "Pearson correlation = ",corr
        print "P-value = ",pvalue
        print "R squared = ",corr**2 #ratio of explained variance to total variance

    else:
        iterable=range(N)
        correlations=[]
        comb=combinations(iterable,2)
        for i in comb:
            (corr,pvalue) = stats.pearsonr(Xflare[:,i[0]],Xflare[:,i[1]])
            correlations.append(corr)

        #sorting the correlations
        correlations=np.array(correlations)
        comb=np.array(comb)
        indices=sorted(range(len(comb)),key=lambda k: correlations[k],reverse=True) #indices of the sorted correlations

        #show the result as a nice plot!
        #we only display the 20 highest correlations
        plt.clf()
        plt.xlabel('Pairs')
        plt.ylabel('Pearson correlation')
        plt.title("20 highest correlations")
        plt.plot(correlations[indices[0:20]],color='r',linestyle='-')
        plt.xticks(range(len(comb[indices[0:20]])),comb[indices[0:20]])
        plt.show()


if __name__ == "__main__":
    main()
