# SVM Classifier for Credit Card Approvals

This script uses a Support Vector Machine (SVM) to classify credit card approval decisions based on various applicant attributes.

Report folder contains:
- [Report](report/Credit_Card_Approvals_Report.pdf): Report of Analyzing Credit Card Approvals Dataset by SVM

The output folder contains:
- [y_pred_pulsar (Pulsar Data)](output/y_pred_pulsar.csv): Predicted results of Pulsar dataset from practice session.
- [Classification Report](output/classification_report.csv): Summary records of each run.
- [Model Performance](output/model_performance.csv): Summary best params, accuracy and confusion matrix of each run.
- [y_pred (Credit Card Approvals Data)](output/y_red.csv): Show approval status of the dataset.
- [Train Set](output/train_set.csv): Training set after preprocessing.
- [Test Set](output/test_set.csv): Test set after preprocessing.
- [Cleaned Data](output/clean_data_from_notebook.csv): Cleaned data by prepocess_raw_data.ipynb.


## Environment
- Python 3.9.6

Install require library:
```bash
pip3 install -r requirements.txt
```

Quick run by shell script [classify.sh](classify.sh)
```bash
./classify.sh

# Execute line below to get permissions to the script before running:
chmod +x classify.sh
```

## Run with Google Colab
You can run this notebook in Google Colab by clicking the button below:

- Practical Session
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1LezrF9_DhVwdnIkCEZX2RT6YkcM2RziU/view?usp=sharing)

- Preprocessing
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1vvBdxLtQdWCcyZddEPGAvbH1OgIX3aDp/view?usp=sharing)

- Credit Card Approvals
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/19tSCIFZSYzyve8tXRX6XcPqsPzDtIpoU/view?usp=sharing)

## [SVM Classifier](svm_classifier.py)

### Arguments

- `--preprocess`: Specify if the dataset whether it is preprocessed. Default is `False`.
- `--data`: Path to the dataset. Default is './data/credit_card_approvals.csv'.
- `--output`: Path to the output directory. Default is './output'.
- `--target`: Target value to classify. Default is 'Approved'.
- `--test_size`: Size of the test set. Default is 0.25.
- `--scaler`: Scaler for the features. Options are "standard", "maxmin", "robust". Default is 'standard'.

### Usage

```bash
python3 svm_classifier.py --preprocess --data <data> --output <output> --target <target> --test_size <test_size> --scaler <scaler>
```

### Example
```bash
python3 svm_classifier.py
```
or
```bash
python3 svm_classifier.py --preprocess --data './data/raw_credit_card_approvals.csv' --output './output' --target 'Approved' --test_size 0.25 --scaler 'standard'
```

## [Preprocessing](preproccessing.py)
This script preprocesses the raw credit card approvals dataset to prepare it for the SVM classifier.
This can be done individually as below and the output of preprocessed data is saved as `preprocessed_credit_card_approvals.csv` inside folder `data` as default.

### Arguments

- `--data`: Path to the raw dataset. Default is './data/raw_credit_card_approvals.csv'.
- `--output`: Path to the output directory where the preprocessed data will be saved. Default is './data'.
- `--target`: Target value to classify. Default is 'Approved'.
- `--test_size`: Size of the test set. Default is 0.25.
- `--scaler`: Scaler for the features. Options are "standard", "maxmin", "robust". Default is 'standard'.

### Usage

```bash
python3 preprocess.py --data <data> --output <output> --target <target> --test_size <test_size> --scaler <scaler>
```

### Example

```bash
python3 preprocess.py
```
or
```bash
python3 preprocess.py --data './data/raw_credit_card_approvals.csv' --output './data' --target 'Approved' --test_size 0.25 --scaler 'standard'
```

## [Fairness Evaluation](fairness.py)

This Python script provides a set of functions to evaluate the fairness of a machine learning model's predictions. It uses various statistical tests and metrics to assess the model's performance across different groups defined by a categorical feature.

### Arguments

- `--feature`: Named feature (column name) for which fairness will be evaluated. (Options: "Gender", "Age", "Debt", "Married", "BankCustomer", "Industry", "Ethnicity", "YearsEmployed", "PriorDefault", "Employed", "CreditScore", "DriversLicense", "Citizen", "Income"). Default is 'Ethnicity'.

### Usage

```bash
python3 fairness.py --feature <feature>
```

### Example
```bash
python3 fairness.py --feature Ethnicity
```