library(dplyr)

data_train <- read.csv("https://raw.githubusercontent.com/thomaspernet/data_csv_r/master/data/titanic_train.csv")

data_test <- read.csv("https://raw.githubusercontent.com/thomaspernet/data_csv_r/master/data/titanic_test.csv")
  
  
library(randomForest)
library(caret)
library(e1071)

## Train the Model

trcontrol <- trainControl(method = "cv", number = 10, search = "grid")

set.seed(1234)

rf_default <- train(survived~.,
                    data = data_train,
                    method = "rf",
                    metric = "Accuracy",
                    trcontrol = trcontrol)

print(rf_default)


## Search best mtry 

set.seed(1234)
tuneGrid <- expand.grid(.mtry = c(1: 10))
# Construct a vector with value from 1:10

rf_mtry <- train(survived~.,
                 data = data_train,
                 method = "rf",
                 metric = "Accuracy",
                 tuneGrid = tuneGrid,
                 trcontrol = trcontrol,
                 importance = TRUE,
                 ntree = 300)

print(rf_mtry)

# The best value of mtry is stored in
best_mtry <- rf_mtry$bestTune$mtry
best_mtry

# For getting the best/max accuracy of mtry
best_acc <- max(rf_mtry$results$Accuracy)
best_acc


## Search the Best maxnodes

store_maxnode <- list()    # results of the model will be stored in this list
tuneGrid <- expand.grid(.mtry = best_mtry)  # use the best value of mtry
for (maxnodes in c(5: 15)) {      #compute the model with the values of maxnodes in that range
  set.seed(1234)
  rf_maxnode <- train(survived~.,
                      data = data_train,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trcontrol = trcontrol,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes, # for each iteration, maxnodes is equal to the current maxnodes.
                      ntree = 300)
  current_iteration <- toString(maxnodes)  # stores as a string variable the value of maxnode.
  store_maxnode[[current_iteration]] <- rf_maxnode # saving the result of the model in the list 
}
results_mtry <- resamples(store_maxnode)  # Arrange the results of the model.
summary(results_mtry)


## search the best ntrees.

store_maxtrees <- list()
for (ntree in c(250, 300, 350, 400, 450, 500, 550, 600, 800, 1000, 2000)) {
  set.seed(5678)
  rf_maxtrees <- train(survived~.,
                       data = data_train,
                       method = "rf",
                       metric = "Accuracy",
                       tuneGrid = tuneGrid,
                       trcontrol = trcontrol,
                       importance = TRUE,
                       nodesize = 14,
                       maxnodes = 24,
                       ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree)


#### final Model ####
# we can train our random forest model with followin parameters

# ntree = 800 - 800 trees will be trained.
# mtry = 3 - 3 feature are chosen for each iteration.
# maxnodes = 11 - maximum 11 nodes in the termoanl node.

fit_rf <- train(survived~.,
                data_train,
                method = "rf",
                metric = "Accuracy",
                tuneGrid = tuneGrid,
                trcontrol = trcontrol,
                importance = TRUE,
                nodesize = 14,
                ntree = 800,
                maxnodes = 11)

summary(fit_rf)


## evaluating the model 

prediction <-predict(fit_rf, data_test)

# confusion matrix 
confusionMatrix(prediction, data_test$survived)

# variable importance 
varImp(fit_rf)
