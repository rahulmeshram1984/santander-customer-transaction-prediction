######## santander-customer transaction prediction 
rm(list=ls())
setwd("D:/Python/santander-customer transaction prediction")
getwd()
### import data ####
train = read.csv(choose.files(), header = T)
test =  read.csv(choose.files(), header = T)

#Loading the libraries
L = c('tidyr','ggplot2','corrgram','usdm','mlbench','caret','lattice','DMwR','rpart','randomForest')

#loading packages
lapply(L, require, character.only = TRUE)
rm(L)
#checking dimensions
dim(train)
dim(test)

#####EXPLORING_DATA#####
#Structure of data
str(train)
str(test)

#Summary of Data
summary(train)
summary(test)

head(train)
######MISSING_VALUES######
#checking for missing values

sum(is.na(train))
sum(is.na(test))
#storing numeric data without ID_code 

train_data=train[,2:202]
View(train_data)
test_data=test[,2:201]
View(test_data)

#factorize the target variable
train_data$target = factor(train_data$target, levels = c(0, 1))

#column names
cnames=colnames(train[,3:202])
cnames

######EXPLORATORY_DATA_ANALYSIS######

#Plot graph for o & 1s in target variable
target = data.frame(table(train$target))
colnames(target) = c("target", "frequency")
ggplot(data=target, aes(x=target, y=frequency, fill=target)) +
  geom_bar(position = 'dodge', stat='identity', alpha=0.5) +theme_classic()


#ploting target distribution in var_0
ggplot(data=train_data, aes_string(x=train_data$var_0,y=train_data$target))+
  geom_point(color="blue",
             fill="#69b3a2",
             shape=21,
             alpha=0.5,
             size=3,
             stroke = 2)+labs(x='var_0',y='target')

#Distribution of data in variables
qqnorm(train_data$var_0)
hist(train_data$var_0)

qqnorm(test_data$var_0)
hist(test_data$var_0)

#####OUTLIER_ANALYSIS######

#Removing outliers from the train set
for (i in cnames) {
  val = train_data[,i][train_data[,i]%in% boxplot.stats(train_data[,i])$out]
  print(length(val))
  train_data[,i][train_data[,i] %in% val] = NA
  
}
#Again checking for missing data after outliers
apply(train_data, 2,function(x) {sum(is.na(x))})

train_data= drop_na(train_data)

#copying data
data_no_outliers =train_data
dim(train_data)

#Removing outliers from the test set
for (i in cnames) {
  val = test_data[,i][test_data[,i]%in% boxplot.stats(test_data[,i])$out]
  print(length(val))
  test_data[,i][test_data[,i] %in% val] = NA
  
}
#Again checking for missing data after outliers
apply(test_data, 2,function(x) {sum(is.na(x))})

test_data= drop_na(test_data)

#copying data
Test_no_outliers =test_data
dim(test_data)

######FEATURE SELECTION #########

#Correlation

correlationMatrix = cor(train_data[,2:201])
correlationMatrix
highlyCorrelated = findCorrelation(correlationMatrix,cutoff = 0.75)
highlyCorrelated

###So we don't have any correlation in the variables so we have to keep all the 200 variables


#######FEATURE_SCALING#######


#normalization on train_data

for(i in cnames){
  print(i)
  train_data[,i] = (train_data[,i] - min(train_data[,i]))/
    (max(train_data[,i] - min(train_data[,i])))
}
View(train_data)


#normalization on test_data

for(i in cnames){
  print(i)
  test_data[,i] = (test_data[,i] - min(test_data[,i]))/
    (max(test_data[,i] - min(test_data[,i])))
}
View(test_data)


#######MODELLING########

library(caret)
#Dividing data into to train and test
set.seed(272)
train_index = createDataPartition(train_data$target,p=0.7,list = FALSE)
train = train_data[train_index,]
test1 = train_data[-train_index,]



#Fitting Logistic Regression to the Training set

logit_model = glm(formula = target ~ .,family = binomial,data = train)
summary(logit_model)
# Predicting the Test set results
log_pred = predict(logit_model, type = 'response', newdata = test1)
y_pred = ifelse(log_pred > 0.5, 1, 0)
y_pred
ConfMatrix_RF = table(test1$target, y_pred)

confusionMatrix(ConfMatrix_RF)


#      y_pred
#   0       1
#0  46740   623
#1  3828    1341

#Accuracy : 0.9153          
#95% CI : (0.9129, 0.9176)
#No Information Rate : 0.9626          
#P-Value [Acc > NIR] : 1               

#Kappa : 0.3403          

#Mcnemar's Test P-Value : <2e-16          

#           Sensitivity : 0.9243          
#          Specificity : 0.6828          
#        Pos Pred Value : 0.9868          
#         Neg Pred Value : 0.2594          
#            Prevalence : 0.9626          
#       Detection Rate : 0.8897          
#  Detection Prevalence : 0.9016          
#     Balanced Accuracy : 0.8035  


#NaiveBayes MOdel

library(caTools)
library(e1071)

#Develop model
NB_model = naiveBayes(target ~ ., data =train)
summary(NB_model)
#predict on test cases 
NB_Pred = predict(NB_model, test1[,2:201], type = "class")
NB_Pred
#Look at confusion matrix
Conf_matrix = table(test1[,1],NB_Pred)
confusionMatrix(Conf_matrix)

##Confusion Matrix and Statistics

#       NB_Pred
#      0     1
# 0 46688   712
# 1  3325  1806


#       NB_Pred
#      0     1
# 0 46688   712
# 1  3325  1806

#           Accuracy : 0.9232          
#             95% CI : (0.9208, 0.9254)
#No Information Rate : 0.9521          
#P-Value [Acc > NIR] : 1               

#              Kappa : 0.4359          

#Mcnemar's Test P-Value : <2e-16          

#            Sensitivity : 0.9335          
#            Specificity : 0.7172          
#         Pos Pred Value : 0.9850          
#         Neg Pred Value : 0.3520          
#             Prevalence : 0.9521          
#         Detection Rate : 0.8888          
#   Detection Prevalence : 0.9023          
#      Balanced Accuracy : 0.8254

#Predict the test_dataset outcome
Test_Pred = predict(NB_model, test_data[,1:200], type = 'class')
Test_Pred
write.csv(Test_Pred, "Test_Pred.csv", row.names = F)




