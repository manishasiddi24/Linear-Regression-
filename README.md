# Linear-Regression-
Linear regression is a supervised machine learning algorithm used to model the relationship between input features and a continuous target variable. In this task, we first import and preprocess the dataset, then split it into training and testing sets. Using Scikit-learn’s LinearRegression, we train the model on the training data.
Steps involved:
1. Import and Preprocess Dataset:
Use either real-world or synthetic data.
Clean and structure it into features (X) and target (y).
2. Split Data:
Divide data into training and testing sets using train_test_split.
3. Train the Model:
Use LinearRegression() from sklearn.linear_model to fit the model on training data.
4. Evaluate the Model:
Predict on test data.
Evaluate using:
MAE (Mean Absolute Error)
MSE (Mean Squared Error)
R² (R-squared Score)
5. Visualize and Interpret:
Plot the regression line (only for simple linear regression).
Understand the coefficient (slope) and intercept, which represent the relationship between features and target.
