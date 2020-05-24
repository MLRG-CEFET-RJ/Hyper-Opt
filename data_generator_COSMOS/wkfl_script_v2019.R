load("RFILE/cosmos.rdata") #variavel cosmos carregada

library(dplyr)
library(smotefamily)
source("Preprocessamento.R")
#source("classifica.R")
#pre_install_libraries() para instalar os pacotes
pre_load_libraries()

#Feature Selection
ugriz <- select(cosmos,U,DU,G,DG,R,DR,I,DI,Z,DZ,class)

#Filter
ugriz_filter <- filter_all(ugriz, all_vars(. > -99)) # -99 filter
ugriz_filter <- filter_all(ugriz, all_vars(. < 99)) #99 filter
ugriz_filter <- filter(ugriz_filter, I >= 14 & I<= 26) #Infrared filter
ugriz_filter <- filter(ugriz_filter, class != 2)

#Sample
ugriz_filter <- sample.stratified(ugriz_filter, "class")
ugriz_train <- as.data.frame(ugriz_filter[[1]])
ugriz_test <- as.data.frame(ugriz_filter[[2]])

#Validation set
ugriz_test_par <- sample.stratified(ugriz_test, "class", perc = 0.5)
ugriz_test <- ugriz_test_par[[1]]
ugriz_val <- ugriz_test_par[[2]]

#Outliers Removal
ugriz_train <- outliers.boxplot(ugriz_train, "class", alpha = 3.0)

#Min-max normalization
pp = preProcess(ugriz_train, method="range")
ugriz_train <- predict(pp, ugriz_train)
ugriz_test <- predict(pp, ugriz_test)
ugriz_val <- predict(pp, ugriz_val)

#SMOTE function
SMOTE_data = SMOTE(ugriz_train[,1:10], ugriz_train$class, K=4, dup_size = 18)
ugriz_train = SMOTE_data$data
ugriz_train = ugriz_train[sample(nrow(ugriz_train), nrow(ugriz_train)), ] #shuffle

write.table(ugriz_train, file="../dataset/cosmos_train_SMOTE.csv", col.names = F, row.names = F, sep = ",", quote = F)
write.table(ugriz_val, file="../dataset/cosmos_val.csv", col.names = F, row.names = F, sep = ",", quote = F)
write.table(ugriz_test, file="../dataset/cosmos_test.csv", col.names = F, row.names = F, sep = ",", quote = F)
