
#Code for model selection for unbalanced classes using -kernlab-, -caret- and -foreach- packages.
#Author: Pablo Campos V.

library(ggplot2)
library(kernlab)
library(caret)

#Let's do it parallel
library(parallel)
detectCores()
library(doParallel)
registerDoParallel(cores=detectCores())
getDoParName()
getDoParWorkers()

#Let's create some fake data
set.seed(123)
x<- rbind(matrix(rnorm(1200), , 2), matrix(rnorm(1200,mean = 3), , 2))
class <- matrix(c(rep(0, 1000), rep(1, 200)))
df<-as.data.frame(cbind(x,class))
df$class<-df$V3
df$V3<-NULL

#Visualization of data
qplot(df$V1,df$V2,color=factor(df$class))


nsampling<-function(base,T){
  
  ################################
  bad<-subset(base,class==1)   #Subset with class equal to one
  num_bad<-nrow(bad)
  good<-subset(base,class==0)  #Subset with class equal to zero
  
  #Loop via Foreach function
  
  aucs <- foreach(i=1:T, .combine=rbind, 
                  .packages=c("kernlab","pROC","caret")) %dopar% {
                    
                    set.seed(i)
                    
                    ind_samp<-sample(1:nrow(good), size=floor(num_bad/2), replace = FALSE, prob = NULL)
                    ind_bad<-sample(1:nrow(bad), size=floor(num_bad/2), replace=FALSE, prob=NULL)
                    
                    samp<-good[ind_samp,]
                    samp_bad<-bad[ind_bad,]
                    samp_complemento<-good[-ind_samp,]
                    samp_bad_complemento<-bad[-ind_bad,]
                    base_complemento<-rbind(samp_complemento,samp_bad_complemento)
                    
                    basesamp<-rbind(samp,samp_bad)
                    sub.base<-basesamp
                    
                    dfx<-sub.base
                    dfx$class<-NULL
                    
                    class<-sub.base$class
                    class[class==1] <- "M"
                    class[class==0] <- "B"
                    class<-as.factor(class)
                    
                    #Parameter Selection using CARET Package
                    fitControl <- trainControl(
                      method = "repeatedcv",
                      number = 10,
                      repeats = 3, classProbs = TRUE, summaryFunction = twoClassSummary)
                    
                    svmGrid <-  expand.grid(C = c(1, 10, 100),
                                            sigma = c(.1,.5,1))
                    
                    set.seed(i)
                    
                    out <- train(as.matrix(dfx), class, method = "svmRadial",
                                 trControl = fitControl, metric = "ROC", tuneGrid = svmGrid)
                    
                    model<-out$finalModel
                    
                    bestC<-out$bestTune$C
                    bestSigma<-out$bestTune$sigma
                    
                    class_test<-base_complemento$class
                    base_complemento$class<-NULL
                    
                    preds_test<-predict(model,newdata=base_complemento,type="probabilities")
                    preds_test<-preds_test[,2]
                    
                    #Computing Area under ROC curve
                    rocobj<-plot.roc(class_test, preds_test,percent=TRUE, 
                                     ci=TRUE,print.auc=TRUE) 
                    
                    output<-c(rocobj$auc[1],model@nSV,bestC,bestSigma)
                    
                  }
  
list_result<-list(aucs)                                           
return(list_result)
}

#CALL TO -NSAMPLING- FUNCTION
#1ST COLUMN: Area under ROC Curve on Test
#2ND COLUMN: Number of support vectors
#3RD COLUMN: Parameter C (Cost)
#4TH COLUMN: Parameter Sigma (Banwidth of RBF Kernel)

result<-nsampling(df,10)
