# Load required packages
install.packages(c("mlbench", "caret", "e1071"))
library(mlbench)
library(caret)
library(e1071)

# Load breast cancer dataset
df <- read.csv("E:/R Final Project/breast_cancer.csv", 
               header = TRUE, sep = ",", na.string = c(""))
head(df)
df <- df[,-1]
# Normalize features

table(df$diagnosis)
df$diagnosis = factor(df$diagnosis, levels =c("B","M"),
                     labels = c("Benign","Malignant"))
#Normalization
normalize = function(x){
  return ((x - min(x))/ (max(x) - min(x)))
}
df1 = as.data.frame(lapply(df[2:30], normalize))
head(df1)


# Combine normalized features with target variable
normalized_df <- data.frame(Class = df$diagnosis, df1)

# Define features (X) and target variable (y)
X <- normalized_df[, -1]
y <- normalized_df$Class

# Split the dataset into training and testing sets
set.seed(569)  # For reproducibility
split_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[split_index, ]
y_train <- y[split_index]
X_test <- X[-split_index, ]
y_test <- y[-split_index]

# Train k-nearest neighbors (knn) model on training data
knn_model <- knn(train = X_train, test = X_test, cl = y_train, k = 23)
table(knn_model)
# Create a confusion matrix for knn model
conf_matrix_knn <- table(Actual = y_test, Predicted = knn_model)
conf_matrix_knn <- confusionMatrix(y_test, knn_model)
# Print confusion matrix
print(conf_matrix_knn)
true_pos_knn <- conf_matrix_knn$table[1, 1]
true_neg_knn <- conf_matrix_knn$table[1, 2] 
false_pos_knn <- conf_matrix_knn$table[2, 1]
false_neg_knn <- conf_matrix_knn$table[2, 2]

recall_knn <- true_pos_knn/(true_pos_knn + false_neg_knn)
precision_cross <- true_pos_knn/(true_pos_knn + false_pos_knn)
print(recall_knn)
print(recall_knn)

#Evaluate the model performance
#confusion Matrix
table(y_test,knn_model)

#Accuracy = (sum of diagonal elements(left to right)/total)*100
((70+38)/113)*100




# Define k-fold cross-validation settings
k_folds <- 10
control <- trainControl(method = "cv", number = k_folds)

# Train k-nearest neighbors (knn) model using k-fold cross-validation
model_cv <- train(Class ~ ., data = normalized_df, method = "knn",
                  trControl = control)

# Predict on the entire dataset using the cross-validated model
predictions_cv <- predict(model_cv, newdata = normalized_df)
table(predictions_cv)

# Create a confusion matrix for k-fold cross-validated knn model
conf_matrix_cv <- confusionMatrix(predictions_cv, normalized_df$Class)
accuracy <- model_cv$results$Accuracy 
mean(accuracy)

# Print confusion matrix from k-fold cross-validation
print(conf_matrix_cv)

#Accuracy = (sum of diagonal elements(left to right)/total)*100
((354+202)/569)*100

true_pos <- conf_matrix_cv$table[1, 1]
true_neg <- conf_matrix_cv$table[1, 2] 
false_pos <- conf_matrix_cv$table[2, 1]
false_neg <- conf_matrix_cv$table[2, 2]

recall_cross <- true_pos/(true_pos + false_neg)
precision_cross <- true_pos/(true_pos + false_pos)
