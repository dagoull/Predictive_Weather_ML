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
#stopCluster(cl)
# Datos Originales, desde fichero de datos...

xx.all<- Data_
preProcValues <- preProcess(xx.all, method = c("center", "scale"))

xx.all <- predict(preProcValues, xx.all)


#w.corr = cor(xx.train[, c(-25, -26)],use="pairwise.complete.obs")
#dim(w.corr)

xx.train$cluster<- as.factor(xx.train$cluster)

#corrplot ( w.corr[13:24,13:24], order="hclust")

## Encontrar las correlaciones altas por parejas
#ind.high.corr = findCorrelation ( w.corr, cutoff = 0.7 )
#length(ind.high.corr)
#ind.high.corr

xx.train<- xx.train[, -c(1,14)]
names(xx.train)[13]<-"Class"
#
# Create the training and test datasets.

# Create the training and test datasets
set.seed(123)

# Step 1: Get row numbers for the training data
trainRowNumbers <- createDataPartition(xx.train$Class, p=0.8, list=FALSE, times = 1)

# Step 2: Create the training  dataset
xx.train <- xx.train[trainRowNumbers,]

# Step 3: Create the test dataset
xx.test <- xx.train[-trainRowNumbers,]

## Preprocesar los datos.
##
#
# preProcValues <- preProcess(xx.train, method = c("center", "scale"))

# xx.train <- predict(preProcValues, xx.train)
# xx.test <- predict(preProcValues, xx.test)

##

# Step 3: Create the test dataset
#xx.test <- xx.train[-trainRowNumbers,]
#names(xx.test)[26]<-"Class"
#xx.test<-  xx.test[, -25]

# Store X and Y for later use.
#x = xx.train[, 1:24]
#y = xx.train$cluster

#xx.train <-xx.train[, -c(13,15,17,19)]
#xx.test<-  xx.test[, -c(13,15,17,19)]

## Balanceo de las clases
set.seed(123)
#xx.train$Class<- as.factor(xx.train$Class)  
xx.train<- upSample(x = xx.train[, -ncol(xx.train)],
                    y = xx.train$Class)
xx.train[, 'Class'] <-  as.factor(xx.train[, 'Class'])
## Segun el analisis 
##xx.train<-  xx.train[, -25]

#xx.train<-  xx.train[, -21]


# Poblar test

set.seed(123)
xx.test<- upSample(x = xx.test[, -ncol(xx.test)],
                   y = xx.test$Class)
xx.test[, 'Class'] <-  as.factor(xx.test[, 'Class'])

# Balanceo de clases con otro metodos
set.seed(123)

xx.train<- downSample(x = xx.train[, -ncol(xx.train)],
                    y = xx.train$Class)
xx.train[, 'Class'] <-  as.factor(xx.train[, 'Class'])


library(DMwR)
set.seed(123)

xx.train<- SMOTE(Class ~ ., data  = xx.train)
xx.train[, 'Class'] <-  as.factor(xx.train[, 'Class'])


D_Mes$Class[D_Mes$max_mm_per_month  > 110] <- "much"
##
##
xx.train <- as.data.frame(xx.train)
xx.train$Class<- as.factor(xx.train$Class)
xx.test <- as.data.frame(xx.test)
xx.test$Class<- as.factor(xx.test$Class)
##

