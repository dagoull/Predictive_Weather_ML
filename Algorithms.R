## Librerias
library(caret)
library(corrplot)
library(mlbench)
library(caretEnsemble)
library(earth)
library(car)
library(corrgram)
library(Hmisc)
library("PerformanceAnalytics")
library(doParallel) #Paralelismo...
parallel::detectCores()

cl <- makePSOCKcluster(2)

registerDoParallel(cl) 
## All subsequent models are then run in parallel


## When you are done:
stopCluster(cl)

#
#

dim(xx.train)
cor(xx.train[, -25],use="pairwise.complete.obs")
#scatterplotMatrix(xx.train[, -25], diagonal = "hist")

w.corr = cor ( xx.train[ , c(- 25, -26, -13, -15, -17, -19) ] ) # Clase de los datos
dim(w.corr)
corrplot ( w.corr[13:20,13:20])
corrplot ( w.corr[13:24,13:24], order="hclust")
corrgram(w.corr[13:24,13:24])

rcorr(w.corr[13:24,13:24]) 

chart.Correlation(w.corr[13:20,13:20], histogram=TRUE, pch=19)
## Encontrar las correlaciones altas por parejas
ind.high.corr = findCorrelation ( w.corr, cutoff = 0.6)
length(ind.high.corr)
ind.high.corr

## Borrar las variables
xx.train = xx.train [ , - ind.high.corr ]
dim(xx.train)


# Metalgoritmos 

#
#
## Pre-processing #https://rpubs.com/Isaac/caret_reg
#-----------------------------------------------------------------------------------------------------
## Control de la Tecnica de Remuestreo: 10-fold CV
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )
set.seed(123)
rf.fit = train ( Class ~ ., data = xx.train,
                 method = "rf",
                 
                 
                 #tuneGrid = rfGrid , 
                 
                 preProc = c("center", "scale", "corr"),
                 #preProc = c("corr"),
                 metric = "Accuracy",
                 #metric = "ROC",
                 tuneLength = 23,
                 
                 trControl = boot.ctrl )
rf.fit
plot(rf.fit,col = "red", main= paste("Random Forest, Accuracy:", round(max(rf.fit$results$Accuracy),2)))

importance <- varImp(rf.fit, scale=FALSE)
plot(importance, main= paste("Random Forest, Accuracy:", round(max(rf.fit$results$Accuracy),2)) )




#
#
#----------------------------------------------------------------------------------------------------------
#
# Máquinas de Soporte Vectorial

#----------------------------------------------------------------------------------------
#Support Vector Machines
#
# polynomial (using kernel = "polydot") and linear (kernel = "vanilladot").
# method values of "svmRadial", "svmLinear", or "svmPoly"
#
set.seed(123)
svmRadial <-caret::train(xx.train [ , -indY ] , xx.train$Class,
                         method = "svmRadial",
                         preProc = c("center", "scale", "corr"),
                         #preProc = c("corr"),
                         tuneLength = 60,
                         trControl = trainControl(method = "cv"),
                         metric ="Accuracy")

plot(svmRadial, main= paste("Support Vector Machines [svmRadial], Acuracy: ", round(max(svmRadial$results$Accuracy),3)))
svmRadial$finalModel

importance <- varImp(svmRadial, scale=FALSE)
plot(importance, main= paste("Support Vector Machines [svmRadial], Accuracy:", round(max(svmRadial$results$Accuracy),2)) )


#"svmRadial"
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )
svmRTuned <-caret::train(Class ~ ., data = xx.train,
                         method = "svmRadial",
                         #preProc = c("center", "scale"),
                         preProc = c("corr"),
                         tuneLength = 20,
                         trControl = boot.ctrl,
                         metric ="Accuracy"
)
#"svmLinear"

svmGrid <- expand.grid(.C = 0.1:20)
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )
svmLinear <-caret::train(Class~ ., data= xx.train,
                         method = "svmLinear",
                         preProc = c("center", "scale", "corr"),
                         #preProc = c("corr"),
                         tuneLength = 30,
                         #tuneGrid = svmGrid,
                         trControl = boot.ctrl,
                         metric ="Accuracy"
)

plot(svmLinear, main= paste("Support Vector Machines [svmLinear], Acuracy: ", round(max(svmLinear$results$Accuracy),3)))
svmLinear$finalModel

#"svmLinear"
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )
svmGrid <- expand.grid(.cost = 1:20)

svmLinear2 <-caret::train(Class~ ., data= xx.train,
                          method = "svmLinear2",
                          preProc = c("center", "scale", "corr"),
                          #preProc = c("corr"),
                          tuneLength = 20,
                          #tuneGrid = svmGrid,
                          trControl = boot.ctrl,
                          metric ="Accuracy"
)

plot(svmLinear2, main= paste("Support Vector Machines [svmLinear2], Acuracy: ", round(max(svmLinear2$results$Accuracy),3)))
svmLinear2$finalModel

##
#"svmRadialCost" Support Vector Machines with Radial Basis Function Kernel
svmGrid <- expand.grid(.C = 1:20)

svmRadialCost <-caret::train(Class~ ., data= xx.train,
                             method = "svmRadialCost",
                             preProc = c("center", "scale", "corr"),
                             #preProc = c("corr"),
                             #tuneLength = 30,
                             tuneGrid = svmGrid,
                             trControl = trainControl(method = "cv"),
                             metric ="Accuracy"
)

plot(svmRadialCost, main= paste("Support Vector Machines [svmRadialCost], Acuracy: ", round(max(svmRadialCost$results$Accuracy),3)))
svmRadialCost$finalModel

##
#"svmPoly"

set.seed(123)

#svmRGrid <- expand.grid(.degree = 1:2, .nprune = 2:38)
svmRGrid <- expand.grid(.degree = 10:50, .scale =c(0.001, 0.1, 0.5, 0.8, 1), .C = c(0.5, 1, 2, 4, 6, 8, 10, 12, 14, 15))
#svmRGrid <- expand.grid(.degree = c(1:4, 20, 22, 25, 30), .scale =c(0.001, 0.01), .C = c(0.1, 4, 6))
#svmRGrid <- expand.grid(.degree = 2, .epsilon = 0.1, .C = 0.5 )

svmPoly <-caret::train(Class ~ ., data = xx.train,
                       method = "svmPoly",
                       preProc = c("center", "scale", "corr"),
                       #preProc = c("corr"),
                       #tuneGrid = svmRGrid,
                       tuneLength = 12,
                       trControl = trainControl(method = "cv"),
                       metric ="Accuracy"
)

svmPoly
trellis.par.set(caretTheme())
plot(svmPoly,  main= paste("Support Vector Machines (svmPoly), Accuracy:", round(max(svmPoly$results$Accuracy),2)))
svmPoly$finalModel
#
#----------------------------------------------------------------------------------------------
#
# xgbLinear
#

boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )

xgb_grid_1 = expand.grid( nrounds = c( 10, 15, 20, 100),
                          
                          
                          lambda = c(0,01, .1, .4, .5, .6, .8),
                          
                          alpha = c(.01, .1, .2, .4, .5, 1),
                          eta = c(.1, .3, .6)) # Ejemplo ampliado
##               eta = .3) # Normal

#xgb_trcontrol_1 = trainControl( method = "cv", number = 3, allowParallel = FALSE )

model_xgbLinear <- train(Class ~ ., data = xx.train, 
                         #tuneGrid = xgb_grid_1, 
                         tuneLength=8,
                         method = "xgbLinear",
                         metric = "Accuracy",
                         preProc = c("center", "scale", "corr"),
                         trControl=boot.ctrl)
model_xgbLinear
trellis.par.set(caretTheme())
plot(model_xgbLinear, main= paste("eXtreme Gradient Boosting (xgbLinear), Accuracy:", round(max(model_xgbLinear$results$Accuracy),2)))

#
#----------------------------------------------------------------------------------------------
#
# Stochastic Gradient Boosting
#
set.seed(123)
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )

gbm_grid = expand.grid( n.trees = c(  20, 100, 500, 1000, 2000),
                        
                        
                        interaction.depth = c(1,2, 4, 6, 8, 10),
                        
                        shrinkage = c(.01, .1, .2, .4, .5, 1),
                        n.minobsinnode = c(.1, .3, .5, .6, .7, .8))

gbm_grid = expand.grid( n.trees = c(1000, 2000, 3000),
                        interaction.depth = c(4, 6,8),  
                        shrinkage= 0.2, n.minobsinnode = c(.1, .3, .5, .6, .7, .8))


model_gbm <- train(Class ~ ., data = xx.train, 
                         tuneGrid = gbm_grid, 
                         #tuneLength=12,
                         method = "gbm",
                         metric = "Accuracy",
                         preProc = c("center", "scale"),
                         trControl=boot.ctrl)
model_gbm
trellis.par.set(caretTheme())
plot(model_gbm, main= paste("Stochastic Gradient Boosting (gbm), Accuracy:", round(max(model_gbm$results$Accuracy),2), "Kappa:", round(max(model_gbm$results$Kappa),2)))

importance <- varImp(model_gbm, scale=FALSE)
plot(importance, main= paste("Stochastic Gradient Boosting (gbm), Accuracy:", round(max(model_gbm$results$Accuracy),2)) )


#
#----------------------------------------------------------------------------------------------
#
#
# Extreme Learning Machine
#

boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )

elm_grid = expand.grid( n.trees = c(  20, 100, 500, 1000),
                        
                        #nhid, actfun
                        interaction.depth = c(1,2, 4, 6),
                        
                        shrinkage = c(.01, .1, .2, .4, .5, 1),
                        n.minobsinnode = c(.1, .3, .6)) 

model_elm <- train(Class ~ ., data = xx.train, 
                   #tuneGrid = gbm_grid, 
                   tuneLength=2,
                   method = "elm",
                   metric = "Accuracy",
                   preProc = c("center", "scale", "corr"),
                   trControl=boot.ctrl)
model_elm
trellis.par.set(caretTheme())
plot(model_elm, main= paste("Extreme Learning Machine (elm), Accuracy:", round(max(model_elm$results$Accuracy),2)))

#importance <- varImp(model_gbm, scale=FALSE)
#
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

model_xgbDART <-caret::train(Class ~ ., data = xx.train,
                             method='xgbDART',
                             metric = "Accuracy",
                             preProc = c("center", "scale", "corr"),
                             #preProc = c("corr"),
                             tuneLength=5, trControl = fitControl, verbose=F)



model_xgbDART
plot(model_xgbDART, main= paste("eXtreme Gradient Boosting (xgbDART), Accuracy:", round(max(model_xgbDART$results$Accuracy),2)))
summary(model_xgbDART)

#----------------------------------------------------------------------------------------------
#
#---------------------------------------------------------------------
# xgbTree
#

boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )

xgb_grid_1 = expand.grid(nrounds= c( 10, 20, 100, 500), max_depth=c(1,2,3), eta= c(.1, .3, .6), 
                         gamma= c(0, 0.01, .1, .4, .5, .6, .8), colsample_bytree = c(0.1,0.5, 0.7), min_child_weight=c(1, 10),
                         subsample=c(0.1, 0.5))

model_xgbTree <- train(Class~ ., data= xx.train, 
                       #tuneGrid = xgb_grid_1, 
                       tuneLength=10,
                       method = "xgbTree",
                       metric = "Accuracy",
                       preProc = c("center", "scale", "corr"),
                       trControl=boot.ctrl)
model_xgbTree
trellis.par.set(caretTheme())
plot(model_xgbTree, main= paste("eXtreme Gradient Boosting (xgbTree), Accuracy:", round(max(model_xgbTree$results$Accuracy),2)))
#-----------------------------------------------------------------------------------------------------------------
#
#
# Generalized Linear Model
#
#-----------------------------------------------------------------------------------------------------
## Control de la Tecnica de Remuestreo: 10-fold CV
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )
set.seed(123)
glm.fit = train ( Class ~ ., data = xx.train,
                 method = "glm",
                 preProc = c("center", "scale", "corr"),
                 #preProc = c("corr"),
                 metric = "Accuracy",
                 #metric = "ROC",
                 #tuneLength = 23,
                 
                 trControl = boot.ctrl )
glm.fit
# plot(glm.fit,col = "red", main= paste("Random Forest, Accuracy:", round(max(glm.fit$results$Accuracy),2)))
importance <- varImp(glm.fit, scale=FALSE)
plot(importance, main= paste("Generalized Linear Model, Accuracy:", round(max(glm.fit$results$Accuracy),2)) )

#-----------------------------------------------------------------------------------------------------------------
#
#
# Generalized Linear Model with Stepwise Feature Selection
#
#-----------------------------------------------------------------------------------------------------
## Control de la Tecnica de Remuestreo: 10-fold CV
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )
set.seed(123)
glmStepAIC.fit = train ( Class ~ ., data = xx.train,
                  method = "glmStepAIC",
                  preProc = c("center", "scale", "corr"),
                  #preProc = c("corr"),
                  metric = "Accuracy",
                  #metric = "ROC",
                  #tuneLength = 23,
                  
                  trControl = boot.ctrl )
glmStepAIC.fit
# plot(glm.fit,col = "red", main= paste("Random Forest, Accuracy:", round(max(glm.fit$results$Accuracy),2)))
importance <- varImp(glm.fit, scale=FALSE)
plot(importance, main= paste("Generalized Linear Model with Stepwise Feature Selection, Accuracy:", round(max(glmStepAIC.fit$results$Accuracy),2)) )

#-----------------------------------------------------------------------------------------------------------------
#
#
# Linear Discriminant Analysis
#
#-----------------------------------------------------------------------------------------------------
## Control de la Tecnica de Remuestreo: 10-fold CV
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )
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

plot(importance, main= paste("Generalized Linear Model with Stepwise Feature Selection, Accuracy:", round(max(lda.fit$results$Accuracy),2)) )

#
#
# Linear Discriminant Analysis
#-----------------------------------------------------------------------------------------------------
## Control de la Tecnica de Remuestreo: 10-fold CV
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )
set.seed(123)
lda2_grid = expand.grid(dimen= c(0.1, 0.5, 0.8, 1, 2, 5, 10, 20, 100))
#dimen (#Discriminant Functions)

lda2.fit = train ( Class ~ ., data = xx.train,
                  method = "lda2",
                  preProc = c("pca"),
                  #preProc = c("corr"),
                  metric = "Accuracy",
                  #metric = "ROC",
                  tuneGrid = lda2_grid, 
                  #tuneLength = 23,
                  
                  trControl = boot.ctrl )
lda2.fit
plot(glm.fit,col = "red", main= paste("Random Forest, Accuracy:", round(max(lda2.fit$results$Accuracy),2)))

plot(importance, main= paste("Generalized Linear Model with Stepwise Feature Selection, Accuracy:", round(max(lda2.fit$results$Accuracy),2)) )


#
#
# Logic Regression
#-----------------------------------------------------------------------------------------------------
## Control de la Tecnica de Remuestreo: 10-fold CV
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )
set.seed(123)
lda2_grid = expand.grid(dimen= c(0.1, 0.5, 0.8, 1, 2, 5, 10, 20, 100))
#dimen (#Discriminant Functions)

lda2.fit = train ( Class ~ ., data = xx.train,
                   method = "lda2",
                   preProc = c("pca"),
                   #preProc = c("corr"),
                   metric = "Accuracy",
                   #metric = "ROC",
                   tuneGrid = lda2_grid, 
                   #tuneLength = 23,
                   
                   trControl = boot.ctrl )
lda2.fit
plot(lda2.fit,col = "red", main= paste("Random Forest, Accuracy:", round(max(lda2.fit$results$Accuracy),2)))

plot(importance, main= paste("Generalized Linear Model with Stepwise Feature Selection, Accuracy:", round(max(lda2.fit$results$Accuracy),2)) )

#
#
# Model Averaged Neural Network
#-----------------------------------------------------------------------------------------------------
## Control de la Tecnica de Remuestreo: 10-fold CV

set.seed(123)
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )
avNNet_grid = expand.grid(size= c(10:25), decay=c(0.0, 0.05, 0.06, 0.08, 0.1, 0.12), bag=TRUE)
#dimen (#Discriminant Functions)

avNNet.fit = train ( Class ~ ., data = xx.train,
                   method = "avNNet",
                   #preProc = c("pca"),
                   preProc = c("center", "scale", "corr"),
                   metric = "Accuracy",
                   #metric = "ROC",
                   tuneGrid = avNNet_grid, 
                  #tuneLength = 6,
                   
                   
                   trControl = boot.ctrl )
avNNet.fit
plot(avNNet.fit, main= paste("Model Averaged Neural Network, Accuracy:", round(max(avNNet.fit$results$Accuracy),2)))

plot(importance, main= paste("Model Averaged Neural Network, Accuracy:", round(max(avNNet.fit$results$Accuracy),2)) )


#
#
# Monotone Multi-Layer Perceptron Neural Network
#-----------------------------------------------------------------------------------------------------
## Control de la Tecnica de Remuestreo: 10-fold CV

set.seed(123)
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )
#monmlp_grid = expand.grid(size= c(1:5), decay=c(0.1, 0.2,0.5, 0.7, 0.8, 0.9), bag=TRUE)
#dimen (#Discriminant Functions)

monmlp.fit = train ( Class ~ ., data = xx.train,
                     method = "monmlp",
                     #preProc = c("pca"),
                     preProc = c("center", "scale", "corr"),
                     metric = "Accuracy",
                     #metric = "ROC",
                     #tuneGrid = monmlp_grid, 
                     tuneLength = 20,
                     
                     
                     trControl = boot.ctrl )
monmlp.fit
plot(monmlp.fit, main= paste("Monotone Multi-Layer Perceptron Neural Network, Accuracy:", round(max(monmlp.fit$results$Accuracy),2)))

plot(importance, main= paste("Monotone Multi-Layer Perceptron Neural Network, Accuracy:", round(max(monmlp.fit$results$Accuracy),2)) )

#
#
# Multi-Layer Perceptron, multiple layers (No tengo claro como funciona)
#-----------------------------------------------------------------------------------------------------
## Control de la Tecnica de Remuestreo: 10-fold CV

set.seed(123)
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )
mlpWeightDecayML_grid = expand.grid(layer1= c(1:10), layer2 =0.1, layer3=0.1, decay=c(0.1, 0.2,0.5, 0.7, 0.8, 0.9))
#dimen (#Discriminant Functions)

mlpWeightDecayML.fit = train ( Class ~ ., data = xx.train,
                     method = "mlpWeightDecayML",
                     #preProc = c("pca"),
                     preProc = c("center", "scale", "corr"),
                     metric = "Accuracy",
                     #metric = "ROC",
                     #tuneGrid = mlpWeightDecayML_grid, 
                     tuneLength = 30,
                     
                     
                     trControl = boot.ctrl )
mlpWeightDecayML.fit
plot(mlpWeightDecayML.fit, main= paste("Multi-Layer Perceptron, multiple layers, Accuracy:", round(max(mlpWeightDecayML.fit$results$Accuracy),2)))

plot(importance, main= paste("Multi-Layer Perceptron, multiple layers, Accuracy:", round(max(mlpWeightDecayML.fit$results$Accuracy),2)) )

#
#
# Multi-Layer Perceptron, multiple layers (No tengo claro como funciona)
#-----------------------------------------------------------------------------------------------------
## Control de la Tecnica de Remuestreo: 10-fold CV

set.seed(123)
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )
mlpML_grid = expand.grid(layer1= c(1:60), layer2 =0, layer3=0 )
#dimen (#Discriminant Functions)

mlpML.fit = train ( Class ~ ., data = xx.train,
                               method = "mlpML",
                               #preProc = c("pca"),
                               preProc = c("center", "scale", "corr"),
                               metric = "Accuracy",
                               #metric = "ROC",
                                tuneGrid = mlpML_grid, 
                               #tuneLength = 30,
                               
                               
                               trControl = boot.ctrl )
mlpML.fit
plot(mlpML.fit, main= paste("Multi-Layer Perceptron, with multiple layers, Accuracy:", round(max(mlpML.fit$results$Accuracy),2)))

plot(importance, main= paste("Multi-Layer Perceptron, with multiple layers, Accuracy:", round(max(mlpML.fit$results$Accuracy),2)) )


#
#Logistic Model Trees
#-----------------------------------------------------------------------------------------------------
## Control de la Tecnica de Remuestreo: 10-fold CV

set.seed(123)
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )
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

plot(importance, main= paste("Logistic Model Trees, with multiple layers, Accuracy:", round(max(LMT.fit$results$Accuracy),2)) )

#
#Multilayer Perceptron Network by Stochastic Gradient Descent
#-----------------------------------------------------------------------------------------------------
## Control de la Tecnica de Remuestreo: 10-fold CV

set.seed(123)
boot.ctrl = trainControl ( method = "cv" , number = 10, classProbs = TRUE )
mlpSGD_grid = expand.grid(size = c(1:5),  l2reg = 0,  lambda =0, learn_rate = 2e-06, momentum = 0.9,  
                          gamma = 0, minibatchsz = 248, repeats = 1 )
#dimen (#Discriminant Functions)

mlpSGD.fit = train ( Class ~ ., data = xx.train,
                  method = "mlpSGD",
                  #preProc = c("pca"),
                  preProc = c("center", "scale", "corr"),
                  metric = "Accuracy",
                  #metric = "ROC",
                  #tuneGrid = LMT_grid, 
                  tuneLength = 12,
                  
                  
                  trControl = boot.ctrl )
mlpSGD.fit
plot(mlpSGD.fit, main= paste("Multilayer Perceptron Network by Stochastic Gradient Descent, Accuracy:", round(max(mlpSGD.fit$results$Accuracy),2)))

plot(importance, main= paste("Multilayer Perceptron Network by Stochastic Gradient Descent, with multiple layers, Accuracy:", round(max(mlpSGD.fit$results$Accuracy),2)) )


#
#-----------------------------------------------------------------------------------------------------------------
# Control de la Tecnica de Remuestreo: 10-fold CV


#-----------------------------------------------------------------------------------------------------------------
#https://rpubs.com/Isaac/caret_reg
#

print(paste("svmLinear2", round(max(svmLinear2$results$Accuracy),3)))
print(paste("svmLinear", round(max(svmLinear$results$Accuracy),3)))
print(paste("svmRadialCost", round(max(svmRadialCost$results$Accuracy),3)))
print(paste("rf", round(max(rf.fit$results$Accuracy),3)))
print(paste("svmRadial", round(max(svmRadial$results$Accuracy),3)))
print(paste("svmPoly", round(max(svmPoly$results$Accuracy),3)))

cat("****** eXtreme Gradient Boosting (xgbLinear). Accuracy:", round(max(model_xgbLinear$results$Accuracy),3), " Kappa:", round(max(model_xgbLinear$results$Kappa),3), "\n")
cat("****** Random Forest (rf). Accuracy:", round(max(model_xgbLinear$results$Accuracy),3), " Kappa:", round(max(model_xgbLinear$results$Kappa),3), "\n")

cat("****** Stochastic Gradient Boosting (gbm) Accuracy:", round(max(model_gbm$results$Accuracy),3), " Kappa:", round(max(model_gbm$results$Kappa),3), "\n")

#cat("****** Support Vector Machines (svmPoly). Accuracy:", round(max(svmPoly$results$Accuracy),2), " Kappa:",round(max(svmPoly$results$Kappa),3)),  "\n")

