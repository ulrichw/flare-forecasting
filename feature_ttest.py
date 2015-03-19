''' Program to perform two-sample t-tests on the active-region features.
Because the sample size is larger than 300, the distribution of the sample mean
can be assumed normal (central limit theorem).
The distribution of the population itself does not need to be normal.
'''

import numpy as np
from   scipy import stats
import sys, getopt

def main():

    #READ COMMAND-LINE ARGUMENTS
    try:
        opts, args = getopt.getopt(sys.argv[1:],"ha:",["help","a="]) #h is for help, a is for the significance level
    except getopt.GetoptError as err:
        print str(err)
        print 'feature_ttest.py -a <significance level>'
        sys.exit(2)

    if len(opts) >= 1:     
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print 'feature_ttest.py -a <significance level>'
                sys.exit()
            elif opt in ("-a", "--a"):
                alpha = float(arg)
            else:
                assert False, "unhandled option"
                sys.exit(2)
    else:
        print 'wrong or missing argument:'
        print 'feature_ttest.py -a <significance level>'
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

    #PERFORM T-TEST
    for i in range(Xflare.shape[1]):
        t, p = stats.ttest_ind(Xflare[:,i],Xnoflare[:,i],equal_var=False) #use Welch's t-test
        print "t-test results for feature %s:" % names[i]
        print "t statistic= %g  p-value = %g" % (t, p)
        if(p<alpha):
            print "the p-value is lower than the significance level, therefore the null hypothesis can be rejected"
        else:
            print "the p-value is larger than the significance level, therefore the null hypothesis cannot be rejected"

if __name__ == '__main__':

    main()
