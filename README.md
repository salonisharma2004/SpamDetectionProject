# Spam Detection System 🔍

Classifies SMS/email messages as spam or legitimate using Machine Learning and NLP techniques.

## Results
- **Accuracy: 85%**
- Dataset: Kaggle SMS Spam Collection (5,574 messages)

## Tech Stack
Python | scikit-learn | NLP | pandas | Matplotlib

## How it works
1. Raw text messages are cleaned and preprocessed
2. TF-IDF vectorization converts text to numerical features
3. Trained classifier predicts spam vs ham
4. Confusion matrix shows model performance

## How to Run
pip install -r requirements.txt
python spam_detection.py

## Sample Output
Input: "Congratulations! You've won a free travel voucher to Paris. Click here to claim."
Output: Is SPAM 
