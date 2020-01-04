startup1 <- read.csv(file.choose())
View(startup1)
library(fastDummies)
startup <- dummy_cols(startup1)
View(startup)

normalize<-function(x){
  return ( (x-min(x))/(max(x)-min(x)))
}
startup_norm<-as.data.frame(lapply(startup[,-c(4)],normalize))
View(startup_norm)
summary(startup$Profit)

smp_size <- floor(0.7*nrow(startup_norm))
set.seed(123)
train_ind <- sample(seq_len(nrow(startup_norm)), size = smp_size)
startup_train <- startup_norm[train_ind, ]
startup_test <- startup_norm[-train_ind, ]

library(neuralnet)
library(nnet)

startup_model <- neuralnet(Profit~.,data=startup_train)
plot(startup_model)
model_results <- compute(startup_model,startup_test[,-4])
cor(startup_test$Profit,model_results$net.result)
plot(startup_test$Profit,model_results$net.result)

startup_model_2 <- neuralnet(Profit~.,data=startup_train,hidden=c(3,2),threshold=0.001)
plot(startup_model_2)
model_results_2 <- compute(startup_model_2,startup_test[,-4])
cor(startup_test$Profit,model_results_2$net.result)
plot(startup_test$Profit,model_results_2$net.result)
