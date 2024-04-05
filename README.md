![1](https://github.com/vivekyannam/Fraud-Detection-In-Banking-Data-Using-Machine-Learning-Algorithms/assets/131403915/a642d26c-ef92-4f15-a69c-80ef9fcc030e)
![image](https://github.com/vivekyannam/Fraud-Detection-In-Banking-Data-Using-Machine-Learning-Algorithms/assets/131403915/fc459f4f-888f-4dfb-b898-ca42fc4664df)
# Fraud Detection System Readme

## Introduction
This repository contains the implementation of an automated banking fraud detection system using machine learning algorithms. The system aims to accurately detect fraudulent transactions in credit card data while providing a user-friendly interface for ease of use. Various machine learning algorithms have been employed and compared to enhance the accuracy of fraud detection.

## Features
- **Data Pre-processing**: The system preprocesses the input data to enhance its quality and prepare it for model training.
- **Model Training**: Utilizes ensemble models including Majority Voting, Adaboost, and Random Forests to train the fraud detection models.
- **Evaluation Metrics**: Performance evaluation is carried out using metrics such as accuracy, precision, recall, Matthews correlation coefficient (MCC), F1-score, and AUC diagrams.
- **Graphical User Interface (GUI)**: A Wxpython-based frontend is implemented to provide a user-friendly interface for interacting with the system.
- **Cross-validation**: Tenfold cross-validation is employed to validate the performance of classifiers.
- **Linear Regression**: A linear regression classifier with thresholding is utilized for its low variance, high accuracy, and lower computation time compared to other classifiers.

## Modules Overview
1. **Ensemble Model**: Implements a combination of machine learning algorithms for efficient credit card fraud transaction detection.
2. **Majority Voting**: Trains a model using the majority voting technique to address data imbalance issues.
3. **Adaboost**: Utilizes Adaptive Boosting (AdaBoost) to enhance the performance of base classifiers.
4. **Random Forests**: Constructs a series of independent decision trees and aggregates their predictions to classify transactions.
5. **Linear Regression**: Employs linear regression classifier with thresholding for accurate fraud detection.

## Usage
1. **Data Collection**: Ensure the dataset is collected from reliable sources as per the requirements of the system.
2. **Setup Environment**: Install required dependencies using VS Code or any preferred Python environment.
3. **Execute Modules**: Run the appropriate modules to preprocess data, train models, and evaluate performance.
4. **Frontend Interaction**: Utilize the Wxpython-based GUI to input details and detect fraudulent transactions.
5. **Testing and Evaluation**: Test the developed system thoroughly and incorporate necessary changes based on evaluation results.
6. **Report Generation**: Generate project reports detailing the methodology, results, and improvements made in fraud detection.

## Conclusion
This project presents a comprehensive solution for banking fraud detection using machine learning techniques. By implementing ensemble models and employing various evaluation metrics, the system aims to provide accurate detection of fraudulent transactions while minimizing false positives. The provided GUI enhances usability, making it accessible for users to interact with the system effectively.

## Contributors
- Yannam Venkata Vivekananda Reddy
- Anish Kumar Patel
- Akshat Kumar


Thank you for your interest in our fraud detection system!

---

If you need any further assistance, don't hesitate to ask!
