concrete <- read.csv(file.choose())
View(concrete)

normalize<-function(x){
  return ( (x-min(x))/(max(x)-min(x)))
}
concrete_norm<-as.data.frame(lapply(concrete,normalize))
View(concrete_norm)
summary(concrete$strength)

concrete_train <- concrete_norm[1:700,]
concrete_test <- concrete_norm[701:1030,]

library(neuralnet)
library(nnet)

concrete_model <- neuralnet(strength~.,data=concrete_train)
plot(concrete_model)
model_results <- compute(concrete_model,concrete_test[1:8])
table(concrete_test$strength,model_results$net.result)
plot(concrete_test$strength,model_results$net.result)

concrete_model_2 <- neuralnet(strength~.,data=concrete_train,hidden=3,stepmax=1e6)
plot(concrete_model_2)
model_results_2 <- compute(concrete_model_2,concrete_test[1:8])
cor(concrete_test$strength,model_results_2$net.result)
plot(concrete_test$strength,model_results_2$net.result)
