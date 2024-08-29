#!/bin/bash

# Preprocess the data
python3 preprocessing.py --data './data/raw_credit_card_approvals.csv' --output './data' --target 'Approved' --test_size 0.25 --scaler 'standard'

# Run the SVM classifier
python3 svm_classifier.py --data './data/preprocessed_credit_card_approvals.csv' --output './output' --target 'Approved' --test_size 0.25 --scaler 'standard'

# Run fairness evaluation
python3 fairness.py