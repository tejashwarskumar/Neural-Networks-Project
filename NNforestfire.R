forestfire1 <- read.csv(file.choose())
View(forestfire1)
library(fastDummies)
forestfires <- dummy_cols(forestfire1$size_category)
forestfire <- cbind(forestfire1,forestfires[,-1])
names(forestfire)[32]<-paste("size_large")
names(forestfire)[33]<-paste("size_small")
View(forestfire)

normalize<-function(x){
  return ( (x-min(x))/(max(x)-min(x)))
}
forestfire_norm<-as.data.frame(lapply(forestfire[-c(1,2,31)],normalize))
View(forestfire_norm)
summary(forestfire$area)

smp_size <- floor(0.8*nrow(forestfire_norm))
set.seed(123)
train_ind <- sample(seq_len(nrow(forestfire_norm)), size = smp_size)
forestfire_train <- forestfire_norm[train_ind, ]
forestfire_test <- forestfire_norm[-train_ind, ]

library(neuralnet)
library(nnet)

forestfire_model <- neuralnet(area~.,data=forestfire_train)
plot(forestfire_model)
model_results <- compute(forestfire_model,forestfire_test[,-9])
cor(forestfire_test$area,model_results$net.result)
plot(forestfire_test$area,model_results$net.result)

forestfire_model_2 <- neuralnet(area~.,data=forestfire_train,hidden=c(3,2),threshold=0.001)
plot(forestfire_model_2)
model_results_2<-compute(forestfire_model_2,forestfire_test[,-9])
cor(forestfire_test$area,model_results_2$net.result)
plot(forestfire_test$area,model_results_2$net.result)
