# Iris Dataset Analysis

This repository contains an analysis of the Iris dataset using Python. The Iris dataset is a well-known dataset often used for testing machine learning algorithms.

## Dataset

- The Iris dataset consists of measurements of sepal and petal dimensions for three species of iris flowers: setosa, versicolor, and virginica.
- We can download the dataset from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data).
- If We prefer a more convenient approach, We can directly load the Iris dataset using the load_iris function from the sklearn.datasets module.

## K-Nearest Neighbors (KNN) Implementation

- We'll use the KNN algorithm to classify the iris species based on their features.
- The Python code for KNN can be found in the knn_iris.py file.

## Deployment with Flask

- To deploy the KNN model, we'll create a Flask web application.
- The app will take input features (sepal length, sepal width, petal length, petal width) and predict the iris species.
- We can find the Flask app in the app.py file.

## Usage

1. Clone this repository.
2. Install the required dependencies (Flask, scikit-learn, pandas, etc.).
3. Run the Flask app: `python app.py`.
4. Access the app in our browser at `http://localhost:5000`.

Feel free to explore and modify the code as needed!
