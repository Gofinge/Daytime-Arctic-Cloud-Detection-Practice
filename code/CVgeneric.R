#########################################################
# Filename:   CVgeneric
# Author:   Gofinge
# Date: 19.4.21
#
# Discription: This is a R file which is mainly used for
# apply cross-validation with genetic classifer.
#
# FunctionList:
# 1. CVgeneric() genetic Crossvalidation
# 2. err_rate(y, y_pred): defaut loss function
########################################################

#### defaut loss function
err_rate <- function(y, y_pred){
  err_rate <- mean(y != y_pred)
  return(err_rate)
}

#### defaut classifier - qda
qdaClassifier <- function(train.x, train.y, valid.x){
  # currFormula <- as.formula(paste(colnames(train.y), '~', paste(colnames(train.x), collapse = '+'), sep = ''))
  fit.qda <- qda(train.x, train.y)
  pred.qda <- predict(fit.qda, valid.x)
  return(list(pred=pred.qda$class, fit=fit.qda))
}

#### defaut classifier - lr
lrClassifier <- function(train.x, train.y, valid.x){
  train.y[train.y == -1] <- 0
  train <- cbind(train.x, train.y)
  currFormula <- as.formula(paste(colnames(train)[ncol(train)], '~', paste(colnames(train.x), collapse = '+'), sep = ''))
  fit.glm <- glm(currFormula, data = train, family=binomial)
  pred.glm <- round(predict(fit.glm,  valid.x, type="response"))
  pred.glm[pred.glm == 0] = -1 
  return(list(pred=pred.glm, fit=fit.glm))
}

# lrClassifier(dplyr::select(train, NDAI, SD, CORR), train$label, dplyr::select(valid, NDAI, SD, CORR))
  
#### genetic CV
CVgeneric <- function(train.x, train.y, test.x = NULL, classifier=lrClassifier, k=10, loss=err_rate){
  require(caret)
  folds <- createFolds(y=train.y, k=k)
  lossList <- rep(0, k)
  for (i in 1: k){
    trainX <- train.x[-folds[[i]],]
    trainY <- train.y[-folds[[i]]]
    testX <- train.x[folds[[i]],]
    testY <- train.y[folds[[i]]]
    
    model <- classifier(trainX, trainY, testX)
    lossList[i] <- loss(testY, model$pred)
  }
  fit.model <- classifier(train.x, train.y, testX)$fit
  
  return(list(lossList = lossList, fit = fit.model))
}

