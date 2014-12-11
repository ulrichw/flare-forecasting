'''
Program to perform a clustering analysis on magnetic-field decay data from solar active regions
3 different active regions (NOAA 11158, 11429, and 12192) were tested
each catalog contains 150 temporal points of magnetic-field decay around a flare time
This code uses the scikit-learn Python module
Two clustering algorithms are run: k-means and DBScan
'''

import time, numpy as np, matplotlib.pyplot as plt, sys, getopt
from sklearn.cluster import DBSCAN, KMeans

def main():

   start = time.clock()
   
   
   #READ COMMAND-LINE ARGUMENTS
   #-----------------------------------------------------------------------------------------------
   
   try:
       opts, args = getopt.getopt(sys.argv[1:],"hc:",["help","choice="]) #h is for help, c is for choice
   except getopt.GetoptError as err:
       print str(err)
       print 'decay_clustering.py -c <choice>'
       sys.exit(2)

   if len(opts) >= 1:     
          for opt, arg in opts:
              if opt in ("-h", "--help"):
                  print 'decay_clustering.py -c <choice>'
                  sys.exit()
              elif opt in ("-c", "--choice"):
                  print arg
                  choice = int(arg)
              else:
                  assert False, "unhandled option"
                  sys.exit(2)
   else:
          print 'wrong or missing argument:'
          print 'decay_clustering.py -c <choice>'
          sys.exit(2)

           
   
   #DECAY CATALOGS FROM XUDONG SUN
   #-----------------------------------------------------------------------------------------------
   
   if choice == 1:
   
      with open('decay_1158.txt','r') as decay_file:
          Xtext=decay_file.readlines()
      decay_file.close()
      Xdecay=[]
      for i in range(len(Xtext)):
          res=Xtext[i].split()
          temp=[]
          for j in range(len(res)):
              temp.append(float(res[j]))
          Xdecay.append(temp)
   
   elif choice == 2:
   
      with open('decay_1429.txt','r') as decay_file:
          Xtext=decay_file.readlines()
      decay_file.close()
      Xdecay=[]
      for i in range(len(Xtext)):
          res=Xtext[i].split()
          temp=[]
          for j in range(len(res)):
              temp.append(float(res[j]))
          Xdecay.append(temp)
   
   elif choice == 3:
   
      with open('decay_2192.txt','r') as decay_file:
          Xtext=decay_file.readlines()
      decay_file.close()
      Xdecay=[]
      for i in range(len(Xtext)):
          res=Xtext[i].split()
          temp=[]
          for j in range(len(res)):
              temp.append(float(res[j]))
          Xdecay.append(temp)
   
   else:
   
       print 'Error: choice must be equal to 1, 2, or 3'
       sys.exit(2)
   
   Xdecay = np.array(Xdecay,dtype=np.float64)
   print 'Catalog characteristics:',Xdecay.shape,Xdecay.dtype,Xdecay.ndim
   
   #K-MEANS ALGORITHM
   #-----------------------------------------------------------------------------------------------
   
   nclusters=3
   kmeans = KMeans(n_clusters=nclusters, init='k-means++', n_init=100, max_iter=500, tol=0.00000001, precompute_distances=True, verbose=0, random_state=None, copy_x=True, n_jobs=1)
   kmeans.fit(Xdecay)
   
   clusters=kmeans.cluster_centers_

   #plot the cluster centers
   plt.title('K-means clustering\n')
   labels = kmeans.labels_
   for i in range(nclusters):
       plt.plot(clusters[i,:])
   print "labels=",set(labels)
   
   plt.show()
   
   #write output in a text file
   if choice == 1:
       filename = 'mylogfile_1158.txt'
   elif choice == 2:
       filename = 'mylogfile_1429.txt'
   elif choice == 3:
       filename = 'mylogfile_2192.txt'
   
   with open(filename,'w') as log_file:
       for i in range(len(labels)):
           log_file.write(str(labels[i])+ '\n')
   log_file.close()
   
   
   #DBSCAN ALGORITHM
   #-----------------------------------------------------------------------------------------------
   
   db = DBSCAN(eps=2.00, min_samples=10).fit(Xdecay)
   labels = db.labels_
   components=db.components_
   
   # Number of clusters in labels, ignoring noise if present.
   nclusters = len(set(labels)) - (1 if -1 in labels else 0) #any example not in a cluster is assigned the label -1 (noise)
   nlab=set(labels)
   
   print('Estimated number of clusters: %d' % nclusters)

   #plot the cluster centers
   plt.title('DBScan\n')
   for i in range(nclusters):
       plt.plot(components[i,:])
   plt.show()
   
   end = time.clock()
   print 'running time=',end-start 

if __name__ == "__main__":
    main()
