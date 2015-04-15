''' a quick implementation of k-means
    using the two-step algorithm     '''

import numpy as np
import matplotlib.pyplot as plt

def kmeans(k,x):
    ''' k is the number of clusters and x is a numpy array
    containing all of the data points'''

    x=np.array(x)
    ndata = x.shape[0] #number of data points
    ndim= x.shape[1]   #number of spatial dimensions 

    min=x.min()
    max=x.max()

    #random assignment of centroids
    np.random.seed(1)
    centroids=np.random.uniform(low=min,high=max,size=(k,ndim))
    assignements=np.empty(ndata,dtype=np.int16)
    centroids_old = np.ones(shape=(k,ndim))

    reldif= np.linalg.norm(centroids-centroids_old)/np.linalg.norm(centroids_old)

    while(reldif > 0.0001):
        
        #assignment step
        for i in range(ndata):
            distance=[]
            for l in range(k):
                distance.append(np.linalg.norm(x[i,:]-centroids[l,:]))
            distance=np.array(distance)
            assignements[i]=np.argmin(distance)
        
        #update step
        centroids_old = np.copy(centroids)
        for l in range(k):
            centroids[l,:]=np.sum(x[assignements == l,:],axis=0)/float(np.sum([assignements == l]))

        #print assignements,np.sum(x[assignements == 0]),np.sum(x[assignements == 1])
        reldif= np.linalg.norm(centroids-centroids_old)/np.linalg.norm(centroids_old)

    return centroids

def main():

    #we test the kmeans algorithm by producing 3 Gaussian clusters
    x1=np.random.normal(loc=(-1,-1),size=(100,2))
    x2=np.random.normal(loc=(3,9),size=(100,2))
    x3=np.random.normal(loc=(3,4),size=(150,2),scale=2)
    x=np.concatenate((x1,x2,x3))

    #we run kmeans and print the results
    res=kmeans(3,x)
    print 'the centroids are:'
    print res

    #we plot the result
    plt.clf()
    plt.plot(*zip(*x1),marker="o",color='r',ls='')
    plt.plot(*zip(*x2),marker="o",color='g',ls='')
    plt.plot(*zip(*x3),marker="o",color='b',ls='')
    plt.plot(*zip(*res),marker="*",color='black',ls='',fillstyle="full",markersize=15)
    plt.show()

if __name__ == "__main__":    
    main()
