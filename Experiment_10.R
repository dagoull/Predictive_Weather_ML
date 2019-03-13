## Librerias
library(caret)
#library(purrr)
library(corrplot)
library(mlbench)
library(caretEnsemble)
library(earth)
library(car)
library(tidyr)
library(dplyr)
library(skimr)
library(gbm)


#
# library(readr)
#xx_train <- read_csv("C:/Users/Dago/Dropbox/Proyecto_Ricardo/Data/xx.train.csv")

#library(radiant.data)
# https://machinelearningmastery.com/linear-discriminant-analysis-for-machine-learning/
# https://sebastianraschka.com/Articles/2014_python_lda.html
#
# https://rpubs.com/Joaquin_AR/226291
#

library(doParallel) #Paralelismo...
parallel::detectCores()

cl <- makePSOCKcluster(3)

registerDoParallel(cl) 

## All subsequent models are then run in parallel
## Copia xx.train
#copia_xx.train <- xx.train 


## Balanceo de las clases

#https://rpubs.com/Joaquin_AR/383283
#
## https://www.analyticsvidhya.com/blog/2016/12/practical-guide-to-implement-machine-learning-with-caret-package-in-r-with-practice-problem/
#
#
# http://www.rebeccabarter.com/blog/2017-11-17-caret_tutorial/
#
#


# https://machinelearningmastery.com/how-to-evaluate-machine-learning-algorithms/
# https://machinelearningmastery.com/how-to-choose-the-right-test-options-when-evaluating-machine-learning-algorithms/
# https://machinelearningmastery.com/randomness-in-machine-learning/
# https://machinelearningmastery.com/how-to-estimate-model-accuracy-in-r-using-the-caret-package/

##
#
#
# https://www.machinelearningplus.com/machine-learning/caret-package/
#
#https://www.machinelearningplus.com/machine-learning/evaluation-metrics-classification-models-r/
#
#https://www.machinelearningplus.com/machine-learning/logistic-regression-tutorial-examples-r/
#
# Logistic Regression - A Complete Tutorial With Examples in R
#
#
# Save And Finalize Your Machine Learning Model in R
# https://machinelearningmastery.com/finalize-machine-learning-models-in-r/


#
# Configuración
#
boot.ctrl = trainControl ( method = "cv" , number = 3, sampling = "up", returnResamp = "final", classProbs = TRUE )

set.seed(123)
t <- proc.time() # Inicia el 
#-----------------------------------------------------------------------------------------------------------------
#
#
# Linear Discriminant Analysis
#
#-----------------------------------------------------------------------------------------------------
set.seed(123)
lda.fit = train ( Class ~ ., data = xx.train,
                  method = "lda",
                  #preProc = c("center", "scale", "corr"),
                  #preProc = c("pca"),
                  preProc = c("corr", 'YeoJohnson'),
                  metric = "Accuracy",
                  #metric = "ROC",
                  tuneLength = 40,
                  
                  trControl = boot.ctrl )
lda.fit

# Compute the confusion matrix
# Step 2: Predict on testData and Compute the confusion matrix
predicted2 <- predict(lda.fit, xx.test)
lda.fit_pred <-confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='dry')



#
#Logistic Model Trees
#-----------------------------------------------------------------------------------------------------

set.seed(123)

#dimen (#Discriminant Functions)

LMT.fit = train ( Class ~ ., data = xx.train,
                  method = "LMT",
                  preProc = c("corr"),
                  #preProc = c("center", "scale", "corr"),
                  metric = "Accuracy",
                  #metric = "ROC",
                  #tuneGrid = LMT_grid, 
                  tuneLength = 60,
                  
                  
                  trControl = boot.ctrl )
LMT.fit

plot(LMT.fit, main= paste("Logistic Model Trees, with multiple layers, Accuracy:", round(max(LMT.fit$results$Accuracy),2)) )

# Compute the confusion matrix

# Step 2: Predict on testData and Compute the confusion matrix
predicted2 <- predict(LMT.fit, xx.test)

LMT.fit_pred <-confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='dry')


#-----------------------------------------------------------------------------------------------------------------
#
#
# Generalized Linear Model
#
#-----------------------------------------------------------------------------------------------------
set.seed(123)
glm.fit = train ( Class ~ ., data = xx.train,
                  method = "glm",
                  #preProc = c("center", "scale", "corr"),
                  preProc = c("corr"),
                  metric = "Accuracy",
                  #metric = "ROC",
                  trControl = boot.ctrl )
glm.fit
#plot(glm.fit,col = "red", main= paste("Random Forest, Accuracy:", round(max(glm.fit$results$Accuracy),2)))
importance <- varImp(glm.fit, scale=FALSE)
plot(importance, main= paste("Generalized Linear Model, Accuracy:", round(max(glm.fit$results$Accuracy),2)) )

# Step 2: Predict on testData and Compute the confusion matrix
predicted <- predict(glm.fit, xx.test)

glm.fit_pred <- confusionMatrix(reference = xx.test$Class, data = predicted, mode='everything', positive='dry')

#
# Maquinas de Soporte Vectorial

#----------------------------------------------------------------------------------------
#Support Vector Machines
#
# polynomial (using kernel = "polydot") and linear (kernel = "vanilladot").
# method values of "svmRadial", "svmLinear", or "svmPoly"

set.seed(123)
##
#"svmPoly"
#svmRGrid <- expand.grid(.degree = 10:50, .scale =c(0.001, 0.1, 0.5, 0.8, 1), .C = c(0.5, 1, 2, 4, 6, 8, 10, 12, 14, 15))
#svmRGrid <- expand.grid(.degree = c(1:4, 20, 22, 25, 30), .scale =c(0.001, 0.01), .C = c(0.1, 4, 6))
svmRGrid <- expand.grid(.degree = 3, .scale = 0.01, .C = 8 )

svmPoly <-caret::train(Class ~ ., data = xx.train,
                       method = "svmPoly",
                       #preProc = c("center", "scale"),
                       preProc = c("corr"),
                       #tuneGrid = svmRGrid,
                       tuneLength = 10,
                       trControl = boot.ctrl,
                       #metric='ROC'
                       metric ="Accuracy"
)

svmPoly
trellis.par.set(caretTheme())
plot(svmPoly,  main= paste("Support Vector Machines (svmPoly), Accuracy:", round(max(svmPoly$results$Accuracy),2), "Kappa: ", round(max(svmPoly$results$Kappa),2)))
svmPoly$finalModel

plot(svmPoly, plotType = "level") # line

predicted2 <- predict(svmPoly, xx.test)

svmPoly_pred <-  confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='dry')



# Metaalgoritmos
#
#arrange(gbm_model$results, RMSE) %>% head
#
#
# http://www.zevross.com/blog/2017/09/19/predictive-modeling-and-machine-learning-in-r-with-the-caret-package/
#----------------------------------------------------------------------------------------------
# Stochastic Gradient Boosting
#
set.seed(123)
gbm_grid = expand.grid( n.trees = c(  20, 100, 500, 1000, 2000),
                        
                        
                        interaction.depth = c(1,2, 4, 6, 8, 10),
                        
                        
                        
                        shrinkage = c(.01, .1, .2, .4, .5, 1),
                        n.minobsinnode = c(.1, .3, .5, .6, .7, .8))

gbm_grid = expand.grid( n.trees = c(1000, 2000, 3000),
                        interaction.depth = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),  
                        shrinkage= c(0.1, 0.2, 0.3), n.minobsinnode = c(0, .1, .3, .5, .6, .7, .8, .9, 1))
model_gbm <- train(Class ~ ., data = xx.train, 
                   #tuneGrid = gbm_grid, 
                   tuneLength=10,
                   method = "gbm",
                   metric = "Accuracy",
                   #preProc = c("center", "scale", "corr"),
                   preProc = c("corr"),
                   trControl=boot.ctrl)
model_gbm
trellis.par.set(caretTheme())
plot(model_gbm, main= paste("Stochastic Gradient Boosting (gbm), Accuracy:", round(max(model_gbm$results$Accuracy),2), "Kappa: ", round(max(model_gbm$results$Kappa),2)))

importance <- varImp(model_gbm, scale=FALSE)
plot(importance, main= paste("Stochastic Gradient Boosting (gbm), Accuracy:", round(max(model_gbm$results$Accuracy),2)) )

predicciones <- predict(model_gbm, xx.test, type = "raw")
model_gbm_pred <- confusionMatrix(predicciones, xx.test$Class)


#
#----------------------------------------------------------------------------------------------------------------
#  rf 
#
#Ramdom Forest

# Metalgoritmos 

#
#
## Pre-processing #https://rpubs.com/Isaac/caret_reg
#-----------------------------------------------------------------------------------------------------
## Control de la Tecnica de Remuestreo: 10-fold CV
set.seed(123)
rf.fit = train ( Class ~ ., data = xx.train,
                 method = "rf",
                 
                 
                 #tuneGrid = rfGrid , 
                 
                 #preProc = c("center", "scale"),
                 preProc = c("corr"),
                 metric = "Accuracy",
                 #metric = "ROC",
                 tuneLength = 30,
                 
                 trControl = boot.ctrl )
rf.fit
plot(rf.fit, main= paste("Random Forest, Accuracy:", round(max(rf.fit$results$Accuracy),2), " Kappa: ", round(max(rf.fit$results$Kappa),2)) )

importance <- varImp(rf.fit, scale=FALSE)
plot(importance, main= paste("Random Forest, Accuracy:", round(max(rf.fit$results$Accuracy),2), " Kappa: ", round(max(rf.fit$results$Kappa),2)) )

predicciones <- predict(rf.fit, xx.test)

rf.fit_pred <-  confusionMatrix(reference = xx.test$Class, data = predicciones, mode='everything', positive='dry')


#
#
## Pre-processing #https://rpubs.com/Isaac/caret_reg
#-----------------------------------------------------------------------------------------------------


#
# k-Nearest Neighbors
#
#-----------------------------------------------------------------------------------------------------
## Control de la Tecnica de Remuestreo: 10-fold CV

set.seed(123)
kknn.fit = train ( Class ~ ., data = xx.train,
                  method = "kknn",
                  #preProc = c("center", "scale", "corr"),
                  #preProc = c("pca"),
                  preProc = c("corr"),
                  metric = "Accuracy",
                  #metric = "ROC",
                  tuneLength = 20,
                  
                  trControl = boot.ctrl )
kknn.fit
plot(kknn.fit, main= paste("k-Nearest Neighbors, Accuracy:", round(max(kknn.fit$results$Accuracy),2)))

importance <- varImp(kknn.fit, scale=FALSE)
plot(importance, main= paste("k-Nearest Neighbors, Accuracy:", round(max(knn.fit$results$Accuracy),2)) )

# Compute the confusion matrix

# Step 2: Predict on testData and Compute the confusion matrix
predicted2 <- predict(kknn.fit, xx.test)

kknn.fit_pred <-  confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='dry')

#---------------------------------------------------------------------------------------------

set.seed(123)
# Hiperpar?metros 
hiperparametros <- expand.grid(mtry = c(2, 5, 10, 15, 21), 
                               min.node.size = c(2, 3, 4, 5, 6:9, 10), 
                               splitrule = "gini")

rf_ranger <- train( Class ~ ., data = xx.train, method = "ranger", tuneGrid = hiperparametros, preProc = c("corr"),
                                      metric = "Accuracy", trControl = boot.ctrl, 
                        # N?mero de ?rboles ajustados 
                        num.trees =2000)
# REPRESENTACI?N GR?FICA # ============================================================================== 
ggplot(rf_ranger, highlight = TRUE) + labs(title = "Evoluci?n del accuracy del modelo Random Forest") + 
  guides(color = guide_legend(title = "mtry"), 
  shape = guide_legend(title = "mtry")) + theme_bw()

predicted2 <- predict(rf_ranger, xx.test)

rf_ranger_pred<- confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='dry')

#--------------------------------------------------------------------------------------------------
# Redes Neronales
# Hiperpar?metros 
set.seed(123)

hiperparametros <- expand.grid(size = c(5, 7, 9, 10, 15, 20, 40, 60, 70), 
                               decay = c(0.01, 0.03, 0.05, 0.08, 0.1))

fit_nnet <- train( Class ~ ., data = xx.train, method = "nnet", tuneGrid = hiperparametros, preProc = c("corr"),
                    metric = "Accuracy", trControl = boot.ctrl, 
                    # Rango de inicializaci?n de los pesos 
                    
                    # N?mero m?ximo de pesos 
                    MaxNWts = 10000, 
                    # Para que no se muestre cada iteracion por pantalla 
                    trace = TRUE )
# REPRESENTACI?N GR?FICA # ============================================================================== 

ggplot(fit_nnet, highlight = TRUE) + 
  labs(title = "Evoluci?n del accuracy del modelo NNET") + theme_bw()

#plot(fit_nnet, rep="best")

predicted2 <- predict(fit_nnet, xx.test)

fit_nnet_pred<- confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='dry')

#
#----------------------------------------------------------------------------------------------------------------
#  xgbDART  
#
#eXtreme Gradient Boosting
#
# Define the training control
set.seed(123)
xgbDART_grid = expand.grid(nrounds = c(50, 100, 200, 300), max_depth = 8, eta = 0.4, gamma = 0, subsample =0.7222222, 
                           colsample_bytree = 0.6, rate_drop = 0.01, skip_drop = 0.95, min_child_weight = 1)


model_xgbDART <-caret::train(Class ~ ., data = xx.train,
                             method='xgbDART',
                             metric = "Accuracy",
                             #preProc = c("center", "scale", "corr"),
                             preProc = c("corr"),
                             tuneLength= 10, 
                             #tuneGrid = xgbDART_grid, 
                             trControl = boot.ctrl, verbose=F)



model_xgbDART
plot(model_xgbDART, main= paste("eXtreme Gradient Boosting (xgbDART), Accuracy:", round(max(model_xgbDART$results$Accuracy),2), "Kappa: ", round(max(model_xgbDART$results$Kappa),2)))
summary(model_xgbDART)

predicciones <- predict(model_xgbDART, xx.test, type = "raw")
model_xgbDART_pred <- confusionMatrix(predicciones, xx.test$Class)


importance <- varImp(model_xgbDART, scale=FALSE)
plot(importance, main= paste("model_xgbDART, Accuracy:", round(max(model_xgbDART$results$Accuracy),2)) )

#
#----------------------------------------------------------------------------------------------------------------
#  xgbLinear  
#
#eXtreme Gradient Boosting
#
# Define the training control
set.seed(123)

xgbLinear_grid = expand.grid(nrounds = c(50, 100, 200, 300), max_depth = 8, eta = 0.4, gamma = 0, subsample =0.7222222, 
                           colsample_bytree = 0.6, rate_drop = 0.01, skip_drop = 0.95, min_child_weight = 1)


model_xgbLinear <-caret::train(Class ~ ., data = xx.train,
                             method='xgbLinear',
                             metric = "Accuracy",
                             #preProc = c("center", "scale", "corr"),
                             preProc = c("corr"),
                             tuneLength= 14, 
                             #tuneGrid = xgbLinear_grid, 
                             trControl = boot.ctrl, verbose=F)



model_xgbLinear
plot(model_xgbLinear, main= paste("eXtreme Gradient Boosting (xgbLinear), Accuracy:", round(max(model_xgbLinear$results$Accuracy),2), "Kappa: ", round(max(model_xgbLinear$results$Kappa),2)))
summary(model_xgbLinear)

predicciones <- predict(model_xgbLinear, xx.test, type = "raw")
model_xgbLinear_pred <- confusionMatrix(predicciones, xx.test$Class)

#
#----------------------------------------------------------------------------------------------------------------
#  xgbTree  
#
#eXtreme Gradient Boosting
#
# Define the training control

set.seed(123)
xgbTree_grid = expand.grid(nrounds = c(50, 100, 200, 300), max_depth = 8, eta = 0.4, gamma = 0, subsample =0.7222222, 
                             colsample_bytree = 0.6, rate_drop = 0.01, skip_drop = 0.95, min_child_weight = 1)


model_xgbTree <-caret::train(Class ~ ., data = xx.train,
                               method='xgbTree',
                               metric = "Accuracy",
                               #preProc = c("center", "scale", "corr"),
                               preProc = c("corr"),
                               tuneLength= 10, 
                               #tuneGrid = xgbTree_grid, 
                               trControl = boot.ctrl, verbose=F)



model_xgbTree
plot(model_xgbTree, main= paste("eXtreme Gradient Boosting (xgbTree), Accuracy:", round(max(model_xgbTree$results$Accuracy),2), "Kappa: ", round(max(model_xgbTree$results$Kappa),2)))
summary(model_xgbTree)

predicciones <- predict(model_xgbTree, xx.test, type = "raw")
model_xgbTree_pred <- confusionMatrix(predicciones, xx.test$Class)


resultados = function(){
  print("Algoritmo --- LMT") 
  print(paste("Accuracy: ", round(max(LMT.fit$results$Accuracy), 2), "Kappa: ", round(max(LMT.fit$results$Kappa),2))) 
  print(LMT.fit_pred$overall)
  print("----------------------------------------------------------------------------------------------------------")
  
  print("Algoritmo --- LDA") 
  print(paste("Accuracy: ", round(max(lda.fit$results$Accuracy), 2), "Kappa: ", round(max(lda.fit$results$Kappa),2))) 
  print(lda.fit_pred$overall)
  print("----------------------------------------------------------------------------------------------------------")

  print("Algoritmo --- kknn") 
  print(paste("Accuracy: ", round(max(kknn.fit$results$Accuracy), 2), "Kappa: ", round(max(kknn.fit$results$Kappa),2))) 
  print(kknn.fit_pred$overall)
  print("----------------------------------------------------------------------------------------------------------")
  
  print("Algoritmo --- RF") 
  print(paste("Accuracy: ", round(max(rf.fit$results$Accuracy), 2), "Kappa: ", round(max(rf.fit$results$Kappa),2))) 
  print(rf.fit_pred$overall) 
  print("----------------------------------------------------------------------------------------------------------")
  
  print("Algoritmo --- RF Ranger") 
  print(paste("Accuracy: ", round(max(rf_ranger$results$Accuracy, na.rm = TRUE), 2), "Kappa: ", round(max(rf_ranger$results$Kappa, na.rm = TRUE),2))) 
  print(rf_ranger_pred$overall)
  print("----------------------------------------------------------------------------------------------------------")
  
  print("Algoritmo --- GLM") 
  print(paste("Accuracy: ", round(max(glm.fit$results$Accuracy), 2), "Kappa: ", round(max(glm.fit$results$Kappa),2))) 
  print(glm.fit_pred$overall)
  print("----------------------------------------------------------------------------------------------------------") 
  
  print("Algoritmo --- GBM") 
  print(paste("Accuracy: ", round(max(model_gbm$results$Accuracy, na.rm = TRUE), 2), "Kappa: ", round(max(model_gbm$results$Kappa, na.rm = TRUE),2))) 
  print(model_gbm_pred$overall)
  print("----------------------------------------------------------------------------------------------------------") 
  
  print("Algoritmo --- xgbDART") 
  print(paste("Accuracy: ", round(max(model_xgbDART$results$Accuracy), 2), "Kappa: ", round(max(model_xgbDART$results$Kappa),2))) 
  print(model_xgbDART_pred$overall)
  print("----------------------------------------------------------------------------------------------------------") 
  
  print("Algoritmo --- xgbTree") 
  print(paste("Accuracy: ", round(max(model_xgbTree$results$Accuracy), 2), "Kappa: ", round(max(model_xgbTree$results$Kappa),2))) 
  print(model_xgbTree_pred$overall)
  print("----------------------------------------------------------------------------------------------------------") 
  
  print("Algoritmo --- xgbLinear") 
  print(paste("Accuracy: ", round(max(model_xgbLinear$results$Accuracy), 2), "Kappa: ", round(max(model_xgbLinear$results$Kappa),2))) 
  print(model_xgbLinear_pred$overall)
  print("----------------------------------------------------------------------------------------------------------") 
  
  print("Algoritmo --- nnet") 
  print(paste("Accuracy: ", round(max(fit_nnet$results$Accuracy), 2), "Kappa: ", round(max(fit_nnet$results$Kappa),2))) 
  print(fit_nnet_pred$overall)
  print("----------------------------------------------------------------------------------------------------------") 
  
  
  print("Algoritmo --- svmPoly") 
  print(paste("Accuracy: ", round(max(svmPoly$results$Accuracy), 2), "Kappa: ", round(max(svmPoly$results$Kappa),2))) 
  print(svmPoly_pred$overall)
  print("----------------------------------------------------------------------------------------------------------") 
  
}

#
#
# Run resamples() to compare the models
#
#, XGBTREE = model_xgbTree, , SVMPOLY = svmPoly,


# Compare model performances using resample()
models_compare <- resamples(list(LMT = LMT.fit, LDA = lda.fit, RF=rf.fit, GBM = model_gbm, NNET = fit_nnet, KKNN = kknn.fit, 
                                 GLM = glm.fit, SVM = svmPoly,
                                 XGBTREE = model_xgbTree, XGBLINEAR = model_xgbLinear, XGBDART=model_xgbDART, RFRANGER = rf_ranger))


# Summary of the models performances
summary(models_compare)

#
# Predicci?n de los modelos 

#mylista<-  list(XGBDART=model_xgbDART, RFRANGER = rf_ranger)


#predict(mylista)

# Draw box plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare,
       aspect = "fill",
       panel = lattice.getOption("panel.bwplot"),
       scales=scales)


# Ejemplos de Graficos

dotplot(models_compare,
        scales =list(x = list(relation = "free")),
        between = list(x = 2))
bwplot(models_compare,
       metric = "Accuracy")

bwplot(models_compare,
       metric = "Kappa")

densityplot(models_compare,
            auto.key = list(columns = 3), metric = "Accuracy",
            pch = "|")


xyplot(models_compare,
       models = c("RF", "GBM"),
       metric = "Accuracy")

splom(models_compare, metric = "Accuracy")
splom(models_compare, variables = "metrics")

parallelplot(models_compare, metric = "Accuracy")

parallelplot(models_compare)

parallelplot(models_compare, metric = "Kappa")


plot(model_xgbTree, metric = "Accuracy", plotType = "level",
     scales = list(x = list(rot = 90)))


#----------------------------------------------------------------------------------------------------
#
#
# Ensembling the predictions
#
#

# Stacking Algorithms - Run multiple algos in one call.
trainControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=10, sampling = "up", 
                             index = createFolds(xx.train$Class, 10),
                             
                             savePredictions = 'final', 
                             classProbs=TRUE)


#algorithmList <- c('LMT.fit', 'lda.fit', 'rf.fit', 'model_gbm', 'fit_nnet', 'kknn.fit', 'glm.fit',
#                   'model_xgbTree', 'model_xgbLinear', 'model_xgbDART', 'rf_ranger')

algorithmList <- c("rf",  "gbm", "glmboost", "nnet", "treebag", "svmLinear", "ranger", "xgbLinear", "svmPoly" )

set.seed(123)
models <- caretList(Class ~ ., data = xx.train, trControl=trainControl, methodList=algorithmList) 
results <- resamples(models)
summary(results)


# Box plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

#
#How to combine the predictions of multiple models to form a final prediction
#
# Create the trainControl
set.seed(123)
stackControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             sampling = "up", 
                             repeats=10,
                             savePredictions = 'final', 
                             classProbs=TRUE)

# Ensemble the predictions of `models` to form a new combined prediction based on glm
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)

# Predict on testData
stack_predicteds <- predict(stack.glm, newdata=xx.test)


head(stack_predicteds)

confusionMatrix(reference = xx.test$Class, data = stack_predicteds, mode='everything', positive='dry')


# Predict
#
# https://rpubs.com/zxs107020/370699
#

#stack_val_preds <- data.frame(predict(stack, val, type = "prob")) data.frame(predict(stack.glm, newdata=xx.test))
#stack_test_preds <- data.frame(predict(stack, test, type = "prob"))


stack_val_preds <- data.frame(predict(stack.glm, xx.test, type = "prob"))

library( randomForest)

MDSplot(rf.fit, xx.train$Class)

## When you are done:
proc.time()-t # Detiene el cronómetro
stopCluster(cl)


# Bibliografi
# https://machinelearningmastery.com/spot-check-machine-learning-algorithms-in-r/

