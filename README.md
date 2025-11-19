# Titanic-Survival-Prediction-API
API for predicting Titanic survival using a machine learning model (Regression)

This is a web service (API) built using the **FastAPI** framework and the **Uvicorn** server to predict the survival probability of Titanic passengers.

The project serves as a demonstration of integrating a Machine Learning model (a Scikit-learn classifier, despite the `.h5` extension, saved via `pickle`) into a modern, asynchronous web application.

## API Functionality

The API provides two main endpoints:

1.  **`/model/predict` (POST):** Accepts a set of passenger features (7 features) and returns the predicted survival status (0 or 1) and the confidence (probability) of that prediction.
2.  **`/model/train` (POST):** Intended for updating or retraining the model with new data (the current implementation verifies file access for future development).

## Technologies & Dependencies

* **Python 3.x**
* **FastAPI:** For building a high-performance, asynchronous API.
* **Uvicorn:** The ASGI server used to run the application.
* **Scikit-learn:** Provides the core ML classification model.
* **NumPy / Pandas:** Used for efficient data handling and formatting.
* **Pydantic:** Used for data validation and defining the API schema.
