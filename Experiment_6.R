## Librerias
library(caret)
library(corrplot)
library(mlbench)
library(caretEnsemble)
library(earth)
library(car)
library(skimr)
library(doParallel) #Paralelismo...
parallel::detectCores()

cl <- makePSOCKcluster(3)

registerDoParallel(cl) 

## All subsequent models are then run in parallel
## Copia xx.train
#copia_xx.train <- xx.train 

## When you are done:
stopCluster(cl)


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


#--------------------------------------------------------------------------------------------------------------------
#
# Gestion de Datos
# Captura datos originales, luego seleccion de los datos training y testing.
# Balanceo de los datos de trainig
# Selecci?n de dos cluster (primero variables locales, luego variables globales)

# Datos Originales, desde fichero de datos...

## variables dummies 
Data_D<- D_Mes 
Data_D$month<- as.factor(Data_D$month)

Data_D <- Data_D[, c(-15)]
dummies <- dummyVars(~ ., data = Data_D)

# The output of predict is a matrix, change it to data frame
Data_D_1 <- data.frame(predict(dummies, newdata = Data_D))

# Datos_Transformados_mes_tmp <-data.frame(Data_D_1, D_Mes[,15]) # D_Mes

Datos_Transformados_mes_tmp <- Data_D_1
Datos_Transformados_mes<- Datos_Transformados_mes_tmp[, -c(1:12)]

xx.all<- Datos_Transformados_mes

preProcValues <- preProcess(xx.all, method = c("center", "scale"))

xx.all <- predict(preProcValues, xx.all)

xx.all <- data.frame(xx.all, Datos_Transformados_mes_tmp[,1:12],  D_Mes[,15])

#w.corr = cor(xx.train[, c(-25, -26)],use="pairwise.complete.obs")

# Quitar mes y lluvia

xx.all<- xx.all[, -c(13)]
#dim(w.corr)

xx.all$Class<- as.factor(xx.all$Class)

#Quitar precipitaciones y la clase

#w.corr = cor(xx.all[, c(-13, -14)],use="pairwise.complete.obs")

#corrplot ( w.corr[1:12,1:12], order="hclust")

#plot.new(); dev.off()

## Encontrar las correlaciones altas por parejas
##ind.high.corr = findCorrelation ( w.corr, cutoff = 0.75 )
##length(ind.high.corr)
##ind.high.corr

## Borrar las variables
##xx.all = xx.all [ , - ind.high.corr ]
##dim(xx.all)

## Borrar las variables
#xx.test = xx.test [ , - ind.high.corr ]
#dim(xx.test)

#names(xx.train)[26]<-"Class"


# Create the training and test datasets
set.seed(123)

# Step 1: Get row numbers for the training data
trainRowNumbers <- createDataPartition(xx.all$Class, p=0.8, list=FALSE)

# Step 2: Create the training  dataset
xx.train <- xx.all[trainRowNumbers,]

# Step 3: Create the test dataset
xx.test <- xx.all[-trainRowNumbers,]

#names(xx.test)[26]<-"Class"
#xx.test<-  xx.test[, -25]

# Store X and Y for later use.
#x = xx.train[, 1:24]
#y = xx.train$cluster

#xx.train <-xx.train[, -c(9)]
#xx.test<-  xx.test[, -c(9)]

## Balanceo de las clases

#xx.train$Class<- as.factor(xx.train$Class)  
xx.train<- upSample(x = xx.train[, -ncol(xx.train)],
                    y = xx.train$cluster)
xx.train[, 'Class'] <-  as.factor(xx.train[, 'Class'])
## Segun el analisis 
##xx.train<-  xx.train[, -25]

xx.train<-  xx.train[, -21]


# Poblar test
xx.test<- upSample(x = xx.test[, -ncol(xx.test)],
                    y = xx.test$Class)
xx.test[, 'Class'] <-  as.factor(xx.test[, 'Class'])

##


#--------------------------------------------------------------------------------------------------------------------
#
# End Gestion de Datos
#
#




skimmed <- skim_to_wide(xx.train)
skimmed[, c(1:5, 9:11, 13, 15:16)]

#  Importance of variables using 
#

featurePlot(x = xx.train[, 1:12], 
            y = xx.train$Class, 
            plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))

# density plots
#
featurePlot(x = xx.train[, 1:8], 
            
            y = xx.train$Class, 
            plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free"))) 
#
#
#---------------------------------------------------------------------------------------------------------------
#
#
# Recursive feature elimination (rfe)
#
#---------------------------------------------------------------------------------------------------------------
set.seed(123)

options(warn=-1)

subsets <- c(1:5,6, 7, 8:24)

ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

lmProfile <- rfe(x=xx.train[, 1:24], y=xx.train$Class,
                 sizes = subsets,
                 rfeControl = ctrl)

lmProfile
plot(lmProfile)



#---------------------------------------------------------------------------------------------------------------
#
# Set the seed for reproducibility
set.seed(123)

# Train the model using randomForest and predict on the training data itself.

#
#Setting up the

# Define the training control
fitControl <- trainControl(
  method = 'cv',                   # k-fold cross validation
  number = 5,                      # number of folds
  savePredictions = 'final',       # saves predictions for optimal tuning parameter
  classProbs = T,                  # should class probabilities be returned
  sampling = "rose", 
  summaryFunction=twoClassSummary  # results summary function
) 

model_mars = train(Class ~ ., data = xx.train, method='earth', trControl = fitControl)
fitted <- predict(model_mars)

model_mars

plot(model_mars, main="Model Accuracies with MARS")

# compute variable importance

varimp_mars <- varImp(model_mars)
plot(varimp_mars, main="Variable Importance with MARS")

#
#---------------------------------------------------------------------------------------------------------------
# Compute the confusion matrix

# Step 2: Predict on testData and Compute the confusion matrix
predicted2 <- predict(model_mars, xx.test)



confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='dry')




#---------------------------------------------------------------------------------------------------------------
#
#Setting up the

# Define the training control
fitControl <- trainControl(
  method = 'cv',                   # k-fold cross validation
  number = 5,                      # number of folds
  savePredictions = 'final',       # saves predictions for optimal tuning parameter
  classProbs = T,                  # should class probabilities be returned
  sampling = "up",               # "rose", "down", "up", "smote" 
  summaryFunction=twoClassSummary  # results summary function
) 

# Step 1: Tune hyper parameters by setting tuneLength
set.seed(123)
model_mars2 = train(Class ~ ., data = xx.train, method='earth', tuneLength = 40,  metric='ROC', trControl = fitControl)
model_mars2
plot(model_mars2)

## Matriz de Confusion
predicted2 <- predict(model_mars2, xx.test)
confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='dry')

#
#Hyper Parameter Tuning using tuneGrid
#

# Step 1: Define the tuneGrid
marsGrid <-  expand.grid(nprune = c(1:24), 
                         degree = c(1:10))

# Step 2: Tune hyper parameters by setting tuneGrid
set.seed(123)
model_mars3 = train(Class ~ ., data = xx.train, method='earth', metric='ROC', tuneGrid = marsGrid, trControl = fitControl)
model_mars3
plot(model_mars3)

## Matriz de Confusion
predicted2 <- predict(model_mars3, xx.test)
confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='dry')

#
#
# Training Adaboost
#
set.seed(123)

# Train the model using adaboost
model_adaboost = train(Class ~ ., data = xx.train, method='adaboost', metric='ROC', tuneLength=8, trControl = fitControl)
model_adaboost
plot(model_adaboost)

## Matriz de Confusion
predicted2 <- predict(model_adaboost, xx.test)
confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='dry')

#
#Training Random Forest
# 

set.seed(123)

# Train the model using rf
model_rf = train(Class ~ ., data = xx.train, method='rf', metric='ROC', tuneLength=25, trControl = fitControl)
model_rf
plot(model_rf)

## Matriz de Confusion
predicted2 <- predict(model_rf, xx.test)
confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='dry')
 
#
#
# xgBoost Dart
#

set.seed(123)

# Train the model using MARS
model_xgbDART = train(Class ~ ., data = xx.train, method='xgbDART',  metric='ROC', tuneLength=10, trControl = fitControl, verbose=F)
model_xgbDART

## Matriz de Confusion
predicted2 <- predict(model_xgbDART, xx.test)
confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='dry')

#
#
# SVM
#

set.seed(123)

# Train the model using MARS
model_svmRadial = train(Class ~ ., data = xx.train,  method='svmRadial',  metric='ROC', tuneLength=25, trControl = fitControl)
model_svmRadial

## Matriz de Confusion
predicted2 <- predict(model_svmRadial, xx.test)
confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='dry')

#
#
# Run resamples() to compare the models
#
#

# Compare model performances using resample()
models_compare <- resamples(list(ADABOOST=model_adaboost, RF=model_rf, XGBDART=model_xgbDART, MARS_2=model_mars2, MARS=model_mars3, SVM=model_svmRadial))

# Summary of the models performances
summary(models_compare)

#
#

# Draw box plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare, scales=scales)


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

#algorithmList <- c('rf', 'adaboost', 'earth', 'xgbDART', 'svmRadial')
algorithmList <- c('svmPoly', 'svmRadial', 'svmLinear', 'svmLinear2', 'svmRadialCost')
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
                             repeats=3,
                             savePredictions = 'final', 
                             classProbs=TRUE)

# Ensemble the predictions of `models` to form a new combined prediction based on glm
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)

# Predict on testData
stack_predicteds <- predict(stack.glm, newdata=xx.test)


head(stack_predicteds)

confusionMatrix(reference = xx.test$Class, data = stack_predicteds, mode='everything', positive='dry')


#
#
#

#-----------------------------------------------------------------------------------------------------------------
#
#
# Linear Discriminant Analysis
#
#-----------------------------------------------------------------------------------------------------
## Control de la Tecnica de Remuestreo: 10-fold CV
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )
boot.ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, sampling = "smote", classProbs = TRUE )
set.seed(123)
lda.fit = train ( Class ~ ., data = xx.train,
                  method = "lda",
                  preProc = c("center", "scale", "corr"),
                  #preProc = c("pca"),
                  #preProc = c("corr"),
                  metric = "Accuracy",
                  #metric = "ROC",
                  #tuneLength = 23,
                  
                  trControl = boot.ctrl )
lda.fit
# plot(glm.fit,col = "red", main= paste("Random Forest, Accuracy:", round(max(glm.fit$results$Accuracy),2)))
importance <- varImp(lda.fit, scale=FALSE)
plot(importance, main= paste("Linear Discriminant Analysis, Accuracy:", round(max(lda.fit$results$Accuracy),2)) )


#-----------------------------------------------------------------------------------------------------
# Compute the confusion matrix

# Step 2: Predict on testData and Compute the confusion matrix
predicted2 <- predict(lda.fit, xx.test)

confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='dry')


#
#Logistic Model Trees
#-----------------------------------------------------------------------------------------------------
## Control de la Tecnica de Remuestreo: 10-fold CV

set.seed(123)
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )
boot.ctrl <- trainControl(method="repeatedcv", number=10, repeats=10, sampling = "smote", classProbs = TRUE )
LMT_grid = expand.grid(layer1= c(1:60), layer2 =0, layer3=0 )
#dimen (#Discriminant Functions)

LMT.fit = train ( Class ~ ., data = xx.train,
                  method = "LMT",
                  #preProc = c("pca"),
                  preProc = c("center", "scale", "corr"),
                  metric = "Accuracy",
                  #metric = "ROC",
                  #tuneGrid = LMT_grid, 
                  tuneLength = 30,
                  
                  
                  trControl = boot.ctrl )
LMT.fit
plot(LMT.fit, main= paste("Logistic Model Trees, Accuracy:", round(max(LMT.fit$results$Accuracy),2)))
importance <- varImp(LMT.fit, scale=FALSE)
plot(importance, main= paste("Logistic Model Trees, with multiple layers, Accuracy:", round(max(LMT.fit$results$Accuracy),2)) )

# Compute the confusion matrix

# Step 2: Predict on testData and Compute the confusion matrix
predicted2 <- predict(LMT.fit, xx.test)

confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='dry')


#-----------------------------------------------------------------------------------------------------------------
#
#
# Generalized Linear Model
#
#-----------------------------------------------------------------------------------------------------
## Control de la Tecnica de Remuestreo: 10-fold CV
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )
boot.ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, classProbs = TRUE )
set.seed(123)
glm.fit = train ( Class ~ ., data = xx.train,
                  method = "glm",
                  preProc = c("center", "scale", "corr"),
                  #preProc = c("corr"),
                  metric = "Accuracy",
                  #metric = "ROC",
                  trControl = boot.ctrl )
glm.fit
#plot(glm.fit,col = "red", main= paste("Random Forest, Accuracy:", round(max(glm.fit$results$Accuracy),2)))
importance <- varImp(glm.fit, scale=FALSE)
plot(importance, main= paste("Generalized Linear Model, Accuracy:", round(max(glm.fit$results$Accuracy),2)) )

#-----------------------------------------------------------------------------------------------------------------
#
#
# No lineales
#
#-----------------------------------------------------------------------------------------------------
#

#
#
#----------------------------------------------------------------------------------------------------------
#
# Maquinas de Soporte Vectorial

#----------------------------------------------------------------------------------------
#Support Vector Machines
#
# polynomial (using kernel = "polydot") and linear (kernel = "vanilladot").
# method values of "svmRadial", "svmLinear", or "svmPoly"
#
set.seed(123)


##
#"svmPoly"

set.seed(123)
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE, sampling = "up" )
boot.ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, classProbs = TRUE )
#svmRGrid <- expand.grid(.degree = 1:2, .nprune = 2:38)
svmRGrid <- expand.grid(.degree = 10:50, .scale =c(0.001, 0.1, 0.5, 0.8, 1), .C = c(0.5, 1, 2, 4, 6, 8, 10, 12, 14, 15))
#svmRGrid <- expand.grid(.degree = c(1:4, 20, 22, 25, 30), .scale =c(0.001, 0.01), .C = c(0.1, 4, 6))
#svmRGrid <- expand.grid(.degree = 2, .epsilon = 0.1, .C = 0.5 )

# Define the training control
boot.ctrl <- trainControl(
  method = 'cv',                   # k-fold cross validation
  number = 5,                      # number of folds
  savePredictions = 'final',       # saves predictions for optimal tuning parameter
  classProbs = T,                  # should class probabilities be returned
  sampling = "up",
  summaryFunction=twoClassSummary  # results summary function
)

svmPoly <-caret::train(Class ~ ., data = xx.train,
                       method = "svmPoly",
                       #preProc = c("center", "scale"),
                       #preProc = c("corr"),
                       #tuneGrid = svmRGrid,
                       tuneLength = 8,
                       trControl = boot.ctrl,
                       metric='ROC'
                       #metric ="Accuracy"
)

svmPoly
trellis.par.set(caretTheme())
plot(svmPoly,  main= paste("Support Vector Machines (svmPoly), Accuracy:", round(max(svmPoly$results$Accuracy),2), "Kappa: ", round(max(svmPoly$results$Kappa),2)))
svmPoly$finalModel

plot(svmPoly, plotType = "line")

predicted2 <- predict(svmPoly, xx.test)

confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='dry')



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
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )
boot.ctrl <- trainControl(method="repeatedcv", number=10, repeats=3)
gbm_grid = expand.grid( n.trees = c(  20, 100, 500, 1000, 2000),
                        
                        
                        interaction.depth = c(1,2, 4, 6, 8, 10),
                        
                        
                        
                        shrinkage = c(.01, .1, .2, .4, .5, 1),
                        n.minobsinnode = c(.1, .3, .5, .6, .7, .8))

gbm_grid = expand.grid( n.trees = c(1000, 2000, 3000),
                        interaction.depth = c(2, 4, 6,8, 10),  
                        shrinkage= 0.2, n.minobsinnode = c(0, .1, .3, .5, .6, .7, .8, .9, 1))
model_gbm <- train(Class ~ ., data = xx.train, 
                   #tuneGrid = gbm_grid, 
                   tuneLength=12,
                   method = "gbm",
                   metric = "Accuracy",
                   preProc = c("center", "scale", "corr"),
                   trControl=boot.ctrl)
model_gbm
trellis.par.set(caretTheme())
plot(model_gbm, main= paste("Stochastic Gradient Boosting (gbm), Accuracy:", round(max(model_gbm$results$Accuracy),2), "Kappa: ", round(max(model_gbm$results$Kappa),2)))

importance <- varImp(model_gbm, scale=FALSE)
plot(importance, main= paste("Stochastic Gradient Boosting (gbm), Accuracy:", round(max(model_gbm$results$Accuracy),2)) )


#
#----------------------------------------------------------------------------------------------------------------
#  xgbDART  
#
#eXtreme Gradient Boosting
#
# Define the training control
set.seed(123)
fitControl <- trainControl(
  method = 'cv',                   # k-fold cross validation
  number = 5,                      # number of folds
  sampling = "up",
  savePredictions = 'final',       # saves predictions for optimal tuning parameter
  classProbs = T,                  # should class probabilities be returned
  summaryFunction=twoClassSummary  # results summary function
  
) 
fitControl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )

xgbDART_grid = expand.grid(nrounds = 250, max_depth = 4, eta = 0.3, gamma = 0, subsample =0.5, 
colsample_bytree = 0.8, rate_drop = 0.01, skip_drop = 0.95, min_child_weight = 1)

model_xgbDART <-caret::train(Class ~ ., data = xx.train,
                             method='xgbDART',
                             metric = "Accuracy",
                             #preProc = c("center", "scale", "corr"),
                             #preProc = c("corr"),
                             tuneLength= 4, 
                             #tuneGrid = xgbDART_grid, 
                             trControl = fitControl, verbose=F)



model_xgbDART
plot(model_xgbDART, main= paste("eXtreme Gradient Boosting (xgbDART), Accuracy:", round(max(model_xgbDART$results$Accuracy),2), "Kappa: ", round(max(model_xgbDART$results$Kappa),2)))
summary(model_xgbDART)

predicciones <- predict(model_xgbDART, xx.test, type = "raw")
confusionMatrix(predicciones, xx.test$Class)

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
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )
boot.ctrl <- trainControl(method="boot", number=1000)

boot.ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, sampling = "up")
set.seed(123)
rf.fit = train ( Class ~ ., data = xx.train,
                 method = "rf",
                 
                 
                 #tuneGrid = rfGrid , 
                 
                 #preProc = c("center", "scale"),
                 preProc = c("corr"),
                 metric = "Accuracy",
                 #metric = "ROC",
                 tuneLength = 23,
                 
                 trControl = boot.ctrl )
rf.fit
plot(rf.fit, main= paste("Random Forest, Accuracy:", round(max(rf.fit$results$Accuracy),2), " Kappa: ", round(max(rf.fit$results$Kappa),2)) )

importance <- varImp(rf.fit, scale=FALSE)
plot(importance, main= paste("Random Forest, Accuracy:", round(max(rf.fit$results$Accuracy),2), " Kappa: ", round(max(rf.fit$results$Kappa),2)) )

predicciones <- predict(rf.fit, xx.test, type = "raw")
confusionMatrix(predicciones, xx.test$Class)

confusionMatrix(reference = xx.test$Class, data = predicciones, mode='everything', positive='dry')


#
#
## Pre-processing #https://rpubs.com/Isaac/caret_reg
#-----------------------------------------------------------------------------------------------------
## Control de la Tecnica de Remuestreo: 10-fold CV
boot.ctrl = trainControl ( method = "cv" , number = 10, sampling = "rose", classProbs = TRUE )
boot.ctrl <- trainControl(method="boot", number=1000)
boot.ctrl <- trainControl(method="repeatedcv", number=10, repeats=3)

boot.ctrl = trainControl ( method = "cv" , number = 10, sampling = "smote", summaryFunction = twoClassSummary, classProbs = TRUE )
set.seed(123)
rf.fit = train ( Class ~ ., data = xx.train,
                 method = "rf",
                 
                 
                 #tuneGrid = rfGrid , 
                 
                 #preProc = c("center", "scale", "corr"),
                 #preProc = c("corr"),
                 #metric = "Accuracy",
                 #metric = "ROC",
                 tuneLength = 19,
                 
                 trControl = boot.ctrl )
rf.fit
plot(rf.fit, main= paste("Random Forest, Accuracy:", round(max(rf.fit$results$Accuracy),2), " Kappa: ", round(max(rf.fit$results$Kappa),2)) )

importance <- varImp(rf.fit, scale=FALSE)
plot(importance, main= paste("Random Forest, Accuracy:", round(max(rf.fit$results$Accuracy),2), " Kappa: ", round(max(rf.fit$results$Kappa),2)) )

predicciones <- predict(rf.fit, xx.test, type = "raw")
confusionMatrix(predicciones, xx.test$Class)

confusionMatrix(reference = xx.test$Class, data = predicciones, mode='everything', positive='dry')
#-------------------------------------------------------------------------------------------------------

## Generate the test set results
#
# http://topepo.github.io/caret/measuring-performance.html#lift-curves
#

# ADABOOST=model_adaboost, RF=model_rf, XGBDART=model_xgbDART, MARS=model_mars3, SVM=model_svmRadial

lift_results <- data.frame(Class = xx.test$Class)
lift_results$rf <- predict(model_rf, xx.test, type = "prob")[,"dry"]
lift_results$xgbDART <- predict(model_xgbDART, xx.test, type = "prob")[,"dry"]
lift_results$mars3 <- predict(model_mars3, xx.test, type = "prob")[,"dry"]
lift_results$svmRadial <- predict(model_svmRadial, xx.test, type = "prob")[,"dry"]
lift_results$adaboost <- predict(model_adaboost, xx.test, type = "prob")[,"dry"]
lift_results$mars2 <- predict(model_mars2, xx.test, type = "prob")[,"dry"]

head(lift_results)

trellis.par.set(caretTheme())
lift_obj <- lift(Class ~ rf+xgbDART+ mars3+ svmRadial+ adaboost + mars2, data = lift_results)
plot(lift_obj, values = 60, auto.key = list(columns = 6,
                                            lines = TRUE,
                                            points = FALSE))
ggplot(lift_obj, values = 60)

#-----------------------------------------------------------------------------

trellis.par.set(caretTheme())
cal_obj <- calibration(Class ~ rf+xgbDART+ mars3+ svmRadial+ adaboost + mars2,
                       data = lift_results,
                       cuts = 13)
plot(cal_obj, type = "l", auto.key = list(columns = 6,
                                          lines = TRUE,
                                          points = FALSE))

ggplot(cal_obj)

#-----------------------------------------------------------------------------

predicciones <- predict(model_adaboost, xx.test, type = "prob")
predicciones_clases <- ifelse(predicciones  > 0.5, "dry", "rainy")
confusionMatrix(predicciones_clases, xx.test$Class)

# ROC Curves
library(caTools)
colAUC(predicciones,xx.test$Class, plotROC = TRUE)

#------------------------------------------------------------------------------
predicciones <- predict(rf.fit, newdata=xx.test, type  = "prob")

colAUC(predicciones, xx.test$Class)

predicciones <- predict(rf.fit, xx.test, type = "prob")
confusionMatrix(predicciones, xx.test$Class)

predicciones <- predict(model_xgbDART, xx.test) 
confusionMatrix(predicciones, xx.test$Class)

# http://rstudio-pubs-static.s3.amazonaws.com/251240_12a8ecea8e144fada41120ddcf52b116.html#comparing-models---extensive-example-and-more-details