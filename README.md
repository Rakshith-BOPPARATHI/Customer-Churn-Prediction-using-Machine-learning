# Customer-Churn-Prediction-using-Machine-learning
The Customer Churn Prediction project employs machine learning to forecast customer churn and guide retention initiatives. Synthetic data mimics telecom customers, with attributes such as contract type, tenure, and charges. The ChurnPredictor class manages data generation, visualization, preprocessing, and model training (Logistic Regression, Random Forest, XGBoost). XGBoost, upon hyperparameter tuning, obtains high accuracy (AUC ~0.85-0.90). Major drivers of churn are short tenure and month-to-month contracts. Recommendations are to target high-risk clients and enhance support quality, with 15-25% churn decrease and substantial revenue savings projected. The pipeline is scalable for actual application.

Objectives:

Create a Predictive Model: Establish a strong machine learning pipeline to precisely predict customers vulnerable to churning based on synthetic telecom-like customer data.
Identify Primary Churn Drivers: Examine feature importance to reveal drivers of customer churn, including contract type, tenure, and support interactions.
Offer Actionable Insights: Create business recommendations for lowering churn rates by targeting specific retention strategies for high-risk customers.
Show Scalability: Build a modular, flexible code base that can be integrated with actual customer data and deployed in production.
Evaluate Model Performance: Compare and contrast several algorithms (Logistic Regression, Random Forest, XGBoost) and tune the top-performing model to obtain high prediction accuracy (e.g., ROC AUC ~0.85-0.90).

Conclusion:

The Customer Churn Prediction Project effectively provides an end-to-end machine learning solution for predicting and preventing customer churn. Through the use of synthetic data and a structured ChurnPredictor class, it obtains high prediction performance with XGBoost (~0.85-0.90 AUC) and detects essential churn drivers such as month-to-month contracts and intensive support calls. The project offers practical suggestions, including customer targeting for high-risk customers and support quality enhancement, and estimates a 15-25% churn decrease and $500K-$1M in lost revenue annually. Its scalable nature provides room for adjustment to suit real-world scenarios, and it is useful for maximizing customer retention and business value.

