

#Author: Pablo Campos V.

library(kernlab)
library(rARPACK)
library(ggplot2)
library(rgl)


#Let's create some fake data
set.seed(123)
x<- rbind(matrix(rnorm(1200), , 2), matrix(rnorm(1200,mean = 3), , 2))
class <- matrix(c(rep(0, 1000), rep(1, 200)))
df<-as.data.frame(cbind(x,class))
df$class<-df$V3
df$V3<-NULL


#Function 
kpca.hand <- function(mat,k,gamma){
  
  #Implementation of a RBF kernel PCA.
  
  #Arguments:
  #  mat: A MxN dataset as matrix class
  # gamma: Parameter for the RBF kernel.
  # k: The number of components to be returned.
  
  #Returns the k eigenvectors (alphas) that correspond to the k largest 
  #eigenvalues (lambdas)
  
  rbf <- rbfdot(sigma = gamma)
  # Computing the MxM kernel matrix.
  kmat<-kernelMatrix(rbf, mat)
  
  # Centering the kernel matrix.
  ones <- matrix(1/dim(kmat)[1],dim(kmat)[1],dim(kmat)[1])
  kmatcent <- kmat - ones%*%kmat - kmat%*%ones + ones%*%kmat%*%ones
  
  # Obtaining eigenvalues in descending order with corresponding 
  # eigenvectors from the centered Kernel matrix.
  # Here we use eigs, which is a wrapper to ARPACK solver for large scale eigenvalue problems.
  eig <- eigs(kmatcent,k)
  
  return(eig)
}

#Example: Fake data  
gamma <- 0.5
kpca.df <- kpca.hand(as.matrix(df),5,gamma)
#Obtaining the i eigenvectors  that corresponds to the k highest eigenvalues (lambdas)
alphas <- kpca.df$vectors
lambdas <- kpca.df$values

#Plot the first two eigenvectors
plot(kpca.df$vectors[,1],kpca.df$vectors[,2])

qplot(kpca.df$vectors[,1],kpca.df$vectors[,2],color=factor(df$class))
supp.df <- as.data.frame(supp.sc)



project <- function(x_new, Xm, gamma, alphas, lambdas){
  #Function to project new data
  dist2 <- function(y){dist(rbind(x_new,y))^2}
  pair.dist <- apply(Xm,1,dist2)
  k.exp <- exp(-gamma*pair.dist)
  
  alphas.norm <- t(alphas)/lambdas
  alphas.norm <- t(alphas.norm)
  
  aux <- k.exp%*%alphas.norm
  return(aux)
}


#Let's create some fake data for test
set.seed(456)
x_n<- rbind(matrix(rnorm(1200), , 2), matrix(rnorm(1200,mean = 3), , 2))
class_n <- matrix(c(rep(0, 1000), rep(1, 200)))
df_n<-as.data.frame(cbind(x_n,class_n))

#Let's call the -project- function over each test data point
proj.iter.test <- apply(as.matrix(df_n),1,project,Xm=as.matrix(df),gamma=gamma,alphas=alphas,
                        lambdas=lambdas)

test.proj <- t(proj.iter.test)

#Plot the projections into the first two components
qplot(test.proj[,1],test.proj[,2],col=as.factor(df_n$V3),
      xlab="1st Principal Component",ylab="2nd Principal Component")+
  geom_point(size=3)

#Plot the projections into the first three components
plot3d(test.proj[,1],test.proj[,2],test.proj[,3],
       size=3,col=as.factor(df_n$V3+1))