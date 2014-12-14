# program to perform K-means and PAM partitioning analyses on magnetic
# field decay data from HMI (see Sun, X. et al., 2015)
# to see if flare footpoints have different signatures from other
# sunspot regions
#------------------------------------------------------------------------

#load relevant libraries
library(cluster)

#read the magnetic field decay data
decay <- read.table('decay_1158.txt')
decay.p <- as.matrix(decay) #get a matrix from the data frame
rm(decay) #to save memory space

decay.p <- na.omit(decay.p) #remove potential NA values
#decay.p <- scale(decay.p)  #scaling of the features (don't use here)

# Determine optimal number of clusters with K-means
# using the within cluster sum of squares
ss <- (nrow(decay.p)-1)*sum(apply(decay.p,2,var))
for (i in 1:8) ss[i] <- sum(kmeans(decay.p,centers=i)$withinss)
plot(1:8,ss,type="l",col="blue",lwd=3,xlab="Number of Clusters",ylab="Within groups sum of squares")

#call the K-means clustering analysis
nclusters <- 3 #number of clusters selected (corresponding to bend in previous plot)
result <- kmeans(decay.p,nclusters,iter.max=400,nstart=10)

# plot the K-means results in a pdf file
pdf('decay_K-means.pdf')
plot(result$centers[1,],type="l",col="red",main="Clusters",ylim=c(-5,5),xlab="features",ylab="decay")
lines(result$centers[2,],col="blue")
lines(result$centers[3,],col="green")
legend(0,5,c("cluster 1","cluster 2","cluster 3"),lty=c(1,1,1),lwd=c(2,2,2),col=c("red","blue","green"),cex=0.5)
dev.off()

#use the PAM (Partitioning Around Medoids) algorithm to partition the data, rather than k-means
result2 <- pam(decay.p,nclusters,FALSE,"euclidian")
print(summary(result2)) #print info on the result of the PAM analysis

# plot the PAM results
pdf('decay_K-PAM.pdf')
plot(result2$medoids[1,],type="l",col="red",main="Medoids",ylim=c(-5,5),xlab="features",ylab="decay")
lines(result2$medoids[2,],col="blue")
lines(result2$medoids[3,],col="green")
legend(0,5,c("cluster 1","cluster 2","cluster 3"),lty=c(1,1,1),lwd=c(2,2,2),col=c("red","blue","green"),cex=0.5)
dev.off()



