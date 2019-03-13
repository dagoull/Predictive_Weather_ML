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

xx.train<- Data_D 
#w.corr = cor(xx.train[, c(-25, -26)],use="pairwise.complete.obs")
dim(w.corr)

xx.train$cluster<- as.factor(xx.train$cluster)

#corrplot ( w.corr[13:24,13:24], order="hclust")

## Encontrar las correlaciones altas por parejas
#ind.high.corr = findCorrelation ( w.corr, cutoff = 0.7 )
#length(ind.high.corr)
#ind.high.corr

#names(xx.train)[26]<-"Class"
#
# Create the training and test datasets.

# Create the training and test datasets
set.seed(123)

# Step 1: Get row numbers for the training data
trainRowNumbers <- createDataPartition(xx.train$cluster, p=0.8, list=FALSE)

# Step 2: Create the training  dataset
xx.train <- xx.train[trainRowNumbers,]

# Step 3: Create the test dataset
xx.test <- xx.train[-trainRowNumbers,]
names(xx.test)[26]<-"Class"
xx.test<-  xx.test[, -25]

# Store X and Y for later use.
x = xx.train[, 1:24]
y = xx.train$cluster

xx.train <-xx.train[, -c(13,15,17,19)]
xx.test<-  xx.test[, -c(13,15,17,19)]

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

featurePlot(x = xx.train[, 13:20], 
            y = xx.train$Class, 
            plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))

# density plots
#
featurePlot(x = xx.train[, 13:20], 
            
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

subsets <- c(1:5, 10, 15, 18, 20, 21)

ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 10,
                   verbose = FALSE)

lmProfile <- rfe(x=xx.train[, 1:13], y=xx.train$Class,
                 sizes = subsets, ntree = 1000,
                 rfeControl = ctrl)

lmProfile



#---------------------------------------------------------------------------------------------------------------
#
# Set the seed for reproducibility
set.seed(123)

# Train the model using randomForest and predict on the training data itself.
model_mars = train(Class ~ ., data = xx.train, method='earth')
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

predicted2 <- predict(svmPoly, xx.test)

confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='cluster1')


#---------------------------------------------------------------------------------------------------------------
#
#Setting up the

# Define the training control
fitControl <- trainControl(
  method = 'cv',                   # k-fold cross validation
  number = 5,                      # number of folds
  savePredictions = 'final',       # saves predictions for optimal tuning parameter
  classProbs = T,                  # should class probabilities be returned
  summaryFunction=twoClassSummary  # results summary function
) 

# Step 1: Tune hyper parameters by setting tuneLength
set.seed(123)
model_mars2 = train(Class ~ ., data = xx.train, method='earth', tuneLength = 40,  metric='ROC', trControl = fitControl)
model_mars2
plot(model_mars2)

## Matriz de Confusion
predicted2 <- predict(model_mars2, xx.test)
confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='cluster2')

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
confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='cluster2')

#
#
# Training Adaboost
#
set.seed(123)

# Train the model using adaboost
model_adaboost = train(Class ~ ., data = xx.train, method='adaboost', metric='ROC', tuneLength=4, trControl = fitControl)
model_adaboost
plot(model_adaboost)

## Matriz de Confusion
predicted2 <- predict(model_adaboost, xx.test)
confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='cluster2')

#
#Training Random Forest
# 

set.seed(123)

# Train the model using rf
model_rf = train(Class ~ ., data = xx.train, method='rf', metric='ROC', tuneLength=20, trControl = fitControl)
model_rf
plot(model_rf)

## Matriz de Confusion
predicted2 <- predict(model_rf, xx.test)
confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='cluster2')

#
#
# xgBoost Dart
#

set.seed(123)

# Train the model using MARS
model_xgbDART = train(Class ~ ., data = xx.train, method='xgbDART', metric='ROC', tuneLength=5, trControl = fitControl, verbose=F)
model_xgbDART

## Matriz de Confusion
predicted2 <- predict(model_xgbDART, xx.test)
confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='cluster2')

#
#
# SVM
#

set.seed(123)

# Train the model using MARS
model_svmRadial = train(Class ~ ., data = xx.train,  method='svmRadial', preProc = c("center", "scale"), metric='ROC', tuneLength=25, trControl = fitControl)
model_svmRadial

## Matriz de Confusion
predicted2 <- predict(model_svmRadial, xx.test)
confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='cluster2')

#
#
# Run resamples() to compare the models
#
#

# Compare model performances using resample()
models_compare <- resamples(list(ADABOOST=model_adaboost, RF=model_rf, XGBDART=model_xgbDART, MARS=model_mars3, SVM=model_svmRadial))

# Summary of the models performances
summary(models_compare)

#
#

# Draw box plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare, scales=scales)

#
#
# Ensembling the predictions
#
#

# Stacking Algorithms - Run multiple algos in one call.
trainControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=10,
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
boot.ctrl <- trainControl(method="repeatedcv", number=10, repeats=3, classProbs = TRUE )
set.seed(123)
lda.fit = train ( Class ~ ., data = xx.train,
                  method = "lda",
                  #preProc = c("center", "scale", "corr"),
                  preProc = c("pca"),
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

confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='cluster2')


#
#Logistic Model Trees
#-----------------------------------------------------------------------------------------------------
## Control de la Tecnica de Remuestreo: 10-fold CV

set.seed(123)
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )
boot.ctrl <- trainControl(method="repeatedcv", number=10, repeats=10, classProbs = TRUE )
LMT_grid = expand.grid(layer1= c(1:60), layer2 =0, layer3=0 )
#dimen (#Discriminant Functions)

LMT.fit = train ( Class ~ ., data = xx.train,
                  method = "LMT",
                  #preProc = c("pca"),
                  #preProc = c("center", "scale"),
                  metric = "Accuracy",
                  #metric = "ROC",
                  #tuneGrid = LMT_grid, 
                  tuneLength = 30,
                  
                  
                  trControl = boot.ctrl )
LMT.fit
plot(LMT.fit, main= paste("Logistic Model Trees, Accuracy:", round(max(LMT.fit$results$Accuracy),2)))
importance <- varImp(LMT.fit, scale=FALSE)
plot(importance, main= paste("Logistic Model Trees, with multiple layers, Accuracy:", round(max(LMT.fit$results$Accuracy),2)) )

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
                  #preProc = c("center", "scale"),
                  #preProc = c("corr"),
                  metric = "Accuracy",
                  #metric = "ROC",
                  trControl = boot.ctrl )
glm.fit
# plot(glm.fit,col = "red", main= paste("Random Forest, Accuracy:", round(max(glm.fit$results$Accuracy),2)))
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
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )
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

confusionMatrix(reference = xx.test$Class, data = predicted2, mode='everything', positive='cluster2')



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
                             #preProc = c("center", "scale"),
                             #preProc = c("corr"),
                             tuneLength=5, 
                             #tuneGrid = xgbDART_grid, 
                             trControl = fitControl, verbose=F)



model_xgbDART
plot(model_xgbDART, main= paste("eXtreme Gradient Boosting (xgbDART), Accuracy:", round(max(model_xgbDART$results$Accuracy),2), "Kappa: ", round(max(model_xgbDART$results$Kappa),2)))
summary(model_xgbDART)

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
boot.ctrl <- trainControl(method="repeatedcv", number=10, repeats=3)
set.seed(123)
rf.fit = train ( Class ~ ., data = xx.train,
                 method = "rf",
                 
                 
                 #tuneGrid = rfGrid , 
                 
                 #preProc = c("center", "scale"),
                 #preProc = c("corr"),
                 metric = "Accuracy",
                 #metric = "ROC",
                 tuneLength = 19,
                 
                 trControl = boot.ctrl )
rf.fit
plot(rf.fit, main= paste("Random Forest, Accuracy:", round(max(rf.fit$results$Accuracy),2), " Kappa: ", round(max(rf.fit$results$Kappa),2)) )

importance <- varImp(rf.fit, scale=FALSE)
plot(importance, main= paste("Random Forest, Accuracy:", round(max(rf.fit$results$Accuracy),2), " Kappa: ", round(max(rf.fit$results$Kappa),2)) )

predicciones <- predict(rf.fit, xx.test)
confusionMatrix(predicciones, xx.test$Class)

confusionMatrix(reference = xx.test$Class, data = predicciones, mode='everything', positive='cluster2')

