# Install necessary packages if they are not already installed
#install.packages("dplyr")
#install.packages("tidyr")
#install.packages("caret")
#install.packages("xgboost")
#install.packages("randomForest")
#install.packages("Metrics")

# Load required libraries
library(dplyr)
library(tidyr)
library(caret)
library(xgboost)
library(randomForest)
library(Metrics)

# Load the dataset
df0 <- read.csv('waze_dataset.csv')

# Display the first five rows of the dataset
head(df0)

# Copy the dataset
df <- df0

# Create 'km_per_driving_day' feature
df <- df %>%
  mutate(km_per_driving_day = driven_km_drives / driving_days) %>%
  mutate(km_per_driving_day = ifelse(is.infinite(km_per_driving_day), 0, km_per_driving_day))

# Create 'percent_sessions_in_last_month' feature
df <- df %>%
  mutate(percent_sessions_in_last_month = sessions / total_sessions)

# Create 'professional_driver' feature
df <- df %>%
  mutate(professional_driver = ifelse(drives >= 60 & driving_days >= 15, 1, 0))

# Create 'total_sessions_per_day' feature
df <- df %>%
  mutate(total_sessions_per_day = total_sessions / n_days_after_onboarding)

# Create 'km_per_hour' feature
df <- df %>%
  mutate(km_per_hour = driven_km_drives / (duration_minutes_drives / 60))

# Create 'km_per_drive' feature
df <- df %>%
  mutate(km_per_drive = driven_km_drives / drives) %>%
  mutate(km_per_drive = ifelse(is.infinite(km_per_drive), 0, km_per_drive))

# Create 'percent_of_drives_to_favorite' feature
df <- df %>%
  mutate(percent_of_drives_to_favorite = (total_navigations_fav1 + total_navigations_fav2) / total_sessions)

# Drop rows with missing values in the 'label' column
df <- df %>%
  drop_na(label)

# Create 'device2' and 'label2' features
df <- df %>%
  mutate(device2 = ifelse(device == 'Android', 0, 1),
         label2 = ifelse(label == 'churned', 1, 0))

# Drop the 'ID' column
df <- df %>%
  select(-ID)

# Display the first five rows of the modified dataset
head(df)


# Split the dataset into training and testing sets
set.seed(1)
trainIndex <- createDataPartition(df$label2, p = .8, 
                                  list = FALSE, 
                                  times = 1)
dfTrain <- df[trainIndex, ]
dfTest  <- df[-trainIndex, ]

# Split the training set into training and validation sets
trainIndex <- createDataPartition(dfTrain$label2, p = .75, 
                                  list = FALSE, 
                                  times = 1)
dfTrain <- dfTrain[trainIndex, ]
dfVal  <- dfTrain[-trainIndex, ]

# Separate features and labels for training
X_train <- dfTrain %>% select(-label, -label2, -device)
y_train <- dfTrain$label2

# Separate features and labels for validation
X_val <- dfVal %>% select(-label, -label2, -device)
y_val <- dfVal$label2

# Separate features and labels for testing
X_test <- dfTest %>% select(-label, -label2, -device)
y_test <- dfTest$label2

# Set up random forest parameters
rf_grid <- expand.grid(
  mtry = floor(sqrt(ncol(X_train))),
  splitrule = "gini",
  min.node.size = 1
)

# Train random forest model
# Ensure the response variable is a factor (classification task)
y_train <- as.factor(y_train)  # Convert y_train to factor

# Define parameter grid
rf_grid <- expand.grid(mtry = c(2, 4, 6), splitrule = "gini", min.node.size = c(1, 5, 10))

# Train random forest model
rf_model <- train(
  X_train, y_train,
  method = "ranger",
  trControl = trainControl(method = "cv", number = 4),
  tuneGrid = rf_grid,
  importance = 'impurity'
)

# Optimal parameters and performance
rf_model$bestTune
rf_model$results

# Predict on validation set
rf_val_preds <- predict(rf_model, X_val)

# Convert factor type to numeric type
y_val_num <- as.numeric(as.character(y_val))
rf_val_preds_num <- as.numeric(as.character(rf_val_preds))

# Calculate RMSE
rf_val_rmse <- rmse(y_val_num, rf_val_preds_num)
rf_val_rmse

# Set up XGBoost parameters
xgb_grid <- expand.grid(
  nrounds = 300,
  max_depth = c(6, 12),
  eta = c(0.01, 0.1),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(3, 5),
  subsample = 1
)

# Train XGBoost model
xgb_model <- train(X_train, y_train,
                   method = "xgbTree",
                   trControl = trainControl(method = "cv", number = 4),
                   tuneGrid = xgb_grid)

# Optimal parameters and performance
xgb_model$bestTune
xgb_model$results

# Predict on validation set
xgb_val_preds <- predict(xgb_model, X_val)

# Convert factor type to numeric type
y_val_num <- as.numeric(as.character(y_val))
xgb_val_preds <- as.numeric(as.character(xgb_val_preds))

# Calculate RMSE
xgb_val_rmse <- rmse(y_val, xgb_val_preds)
xgb_val_rmse

# Calculate accuracy
accuracy <- mean(y_val == xgb_val_preds)
accuracy
