"""_summary_: This script is used to classify the dataset by SVM.
Author: 
- Quang Huy Phung
- Dinh Minh Nguyen 
- Luong Phuong Truc Huynh
Date: 2024-04-13
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from log import Logger
from preprocessing import Preprocessor

class SVMClassifier:
    def __init__(self, args):
        super(SVMClassifier, self).__init__()
        
        self.data_path = args.data
        self.preprocess = args.preprocess
        self.output_dir = args.output
        self.target_feature = args.target
        self.test_size = args.test_size
        self.scaler_type = args.scaler
        self.scaler = self.init_scaler(self.scaler_type)
        
        self.logger = Logger(self.output_dir)
        self.preprocessor = Preprocessor()

    @staticmethod
    def label_encode_categorical(data):
        """
        Converts categorical data into numerical form using LabelEncoder. 
        This is necessary because machine learning algorithms typically work with numerical data.

        Args:
            data (pd.DataFrame): The data to be encoded.

        Returns:
            pd.DataFrame: The data with categorical variables encoded as numerical values.
        """
        le = LabelEncoder()
        mappings = {}
        
        for col in data.columns:
            if data[col].dtypes=='object':
                data[col]=le.fit_transform(data[col])
                mappings[col] = le.classes_
        return data, mappings
    
    @staticmethod
    def label_decode_numerical(data, mappings):
        le = LabelEncoder()
        for col, classes in mappings.items():
            le.classes_ = classes
            data[col] = le.inverse_transform(data[col].astype(int))

        return data

    @staticmethod
    def init_scaler(scaler):
        """
        Initializes the scaler based on the provided type. 

        Args:
            scaler (str): The type of scaler to use. Options are 'standard', 'maxmin', and 'robust'.

        Returns:
            Scaler: The initialized scaler. Default is StandardScaler if the provided type is not recognized.
        """
        scaler_map = {
            'standard': StandardScaler(),
            'maxmin': MinMaxScaler(),
            'robust': RobustScaler(),
        }
        return scaler_map.get(scaler, StandardScaler())
    
    @staticmethod
    def tune_hyperparameter():
        """
        Tunes the hyperparameters of the SVM using GridSearchCV. 

        Returns:
            GridSearchCV: The GridSearchCV object after fitting. This object can be used to access the best parameters found.
        """
        param_grid = {
            'C': np.linspace(2 ** -5, 2 ** 11, 32),
            'kernel': ['rbf'],
            'gamma': np.linspace(2 ** -15, 2 ** -5, 32)
        }
        model_clf = svm.SVC()
        grid = GridSearchCV(model_clf, param_grid, refit = True, verbose = 3)
        return grid
    
    def preprocess_credit_card_approvals_dataset(self, data):
        """
        This function preprocesses the credit card approvals dataset. It performs several steps:
        
        1. Renames the columns of the dataset.
        2. Drops the 'ZipCode' column.
        3. Replaces missing values in the dataset.
        4. Handles continuous variables by applying appropriate transformations.
        5. Calculates missing values for categorical variables and selects features with no missing values.
        6. Label encodes the categorical variables.
        7. Handles categorical variables by applying appropriate transformations.
        8. Saves the preprocessed data to a CSV file.

        Args:
            data (pandas.DataFrame): The DataFrame to be preprocessed.

        Returns:
            pandas.DataFrame: The preprocessed DataFrame.
        """
        data = self.preprocessor.rename_columns(data, ['Gender', 'Age', 'Debt', 'Married', 'BankCustomer', 'Industry', 'Ethnicity', 'YearsEmployed', 'PriorDefault', 'Employed', 'CreditScore', 'DriversLicense', 'Citizen', 'ZipCode', 'Income', 'Approved'])
        data.drop('ZipCode', axis=1, inplace=True)
        
        data = self.preprocessor.replace_missing_values(data)
        
        continuous_variables = ['Age', 'Debt', 'YearsEmployed', 'CreditScore', 'Income']
        data = self.preprocessor.handle_continuous_values(data, continuous_variables)
        
        categorical_variables = ['Gender', 'Married', 'BankCustomer', 'Industry', 'Ethnicity', 'PriorDefault', 'Employed', 'DriversLicense', 'Citizen', 'Approved']
        mv = self.preprocessor.calculate_missing_values(data[categorical_variables])
        no_missing_data = mv[mv['NumberMissing'] == 0]
        feature_names = no_missing_data['Feature'].values

        cleaned_df = data[feature_names]
        cleaned_df = self.preprocessor.label_encode_categorical(cleaned_df)
        data.update(cleaned_df)

        data = self.preprocessor.handle_categorical_variables(data, categorical_variables, feature_names)
        
        self.preprocessor.save_to_csv(data)
        
        return data
    
    def prepare_data_train(self, data):
        """
        Pre-processes the data by removing duplicates, encoding categorical variables, handling missing values, 
        splitting the data into training and test sets, and scaling the features.
        
        Args:
            data (pd.DataFrame): The data to be preprocessed.

        Returns:
            tuple: A tuple containing the training and test sets for the features (X) and the target variable (y).
        """
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Convert categorical data into numerical form

        data, mappings = self.label_encode_categorical(data)
        
        X = data.drop(self.target_feature, axis=1)
        y = data[self.target_feature]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)

        # Save train set
        train_set = np.hstack([X_train, y_train.to_numpy().reshape(-1, 1)])
        train_data = pd.DataFrame(train_set, columns=data.columns)
        train_data = self.label_decode_numerical(train_data, mappings)
        train_data.to_csv(os.path.join(self.output_dir, 'train_set.csv'), index=False)

        # Save test set
        test_set = np.hstack([X_test, y_test.to_numpy().reshape(-1, 1)])
        test_data = pd.DataFrame(test_set, columns=data.columns)
        test_data = self.label_decode_numerical(test_data, mappings)
        test_data.to_csv(os.path.join(self.output_dir, 'test_set.csv'), index=False)

        # Handle missing values
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(data.drop(self.target_feature, axis=1).to_numpy())
        # data = pd.DataFrame(imp.fit_transform(data), columns = data.columns)

        # Normalize the data
        X_train = imp.transform(X_train)
        X_train = self.scaler.fit_transform(X_train)
        X_test = imp.transform(X_test)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test

    def train(self, model, X_train, y_train):
        """
        Trains the provided model using the training data.

        Args:
            model (sklearn estimator): The machine learning model to be trained.
            X_train (pd.DataFrame): The features of the training set.
            y_train (pd.Series): The target variable of the training set.

        Returns:
            sklearn estimator: The trained model.
        """
        model.fit(X_train, y_train)
        return model

    def evaluate(self, model, y_test, y_pred):
        """
        Evaluates the provided model by calculating its accuracy on the test set.

        Args:
            model (sklearn estimator): The machine learning model to be evaluated.
            y_test (pd.Series): The target variable of the test set.
            y_pred (np.array): The predictions made by the model on the test set.

        Returns:
            float: The accuracy of the model on the test set.
        """
        print("Best parameters found: ", model.best_params_, "\n")
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}", "\n")
        
        clf_report = classification_report(y_test, y_pred, output_dict=True)
        clf_report_df = pd.DataFrame(clf_report).transpose()
        print(f"Classification report: \n{clf_report_df}", "\n")
        
        conf_mat = confusion_matrix(y_test, y_pred,)
        tn, fp, fn, tp = conf_mat.ravel()
        
        print("Confusion matrix:")
        print(f"| TP | FP |   | {tp:2d} | {fp:2d} |")
        print(f"|----|----| = |----|----|")
        print(f"| FN | TN |   | {fn:2d} | {tn:2d} |")
        
        self.logger.log(model.best_params_, accuracy, conf_mat, clf_report_df)
        
        return accuracy, clf_report, conf_mat

    def predict(self, model, test_data):
        """
        Makes predictions on the test set using the provided model.

        Args:
            model (sklearn estimator): The machine learning model to make predictions with.
            test_data (pd.DataFrame): The test set to make predictions on.

        Returns:
            np.array: The predictions made by the model on the test set.
        """
        return model.predict(test_data)

    def save_results(self, pred_results):
        """
        Saves the provided results to a CSV file in the specified output directory.

        Args:
            pred_results (np.array): The predictions made by the model. 
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Save predicts
        y_pred = pd.DataFrame(pred_results, columns=[self.target_feature])
        y_pred.to_csv(os.path.join(self.output_dir, 'y_pred.csv'), index=False)

    def run(self):
        data = pd.read_csv(self.data_path)
        
        if self.preprocess:
            data = self.preprocess_credit_card_approvals_dataset(data)
        X_train, X_test, y_train, y_test = self.prepare_data_train(data)
        
        grid = self.tune_hyperparameter()
        model = self.train(grid, X_train, y_train)
        
        best_model = model.best_estimator_
        best_model = self.train(best_model, X_train, y_train)
        
        y_pred = self.predict(best_model, X_test)
        
        self.evaluate(model, y_test, y_pred)
        
        self.save_results(y_pred)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify by SVM')
    parser.add_argument('--preprocess', action='store_true', 
                        help='Specify to preprocess the dataset.')
    parser.add_argument('--data', type=str, default='./data/credit_card_approvals.csv',
                        help='Path to the dataset.')
    parser.add_argument('--output', type=str, default='./output',
                        help='Path to the output directory.')
    parser.add_argument('--target', type=str, default='Approved',
                        help='Target value to classify.')
    parser.add_argument('--test_size', type=float, default=0.25,
                        help='Size of the test set.')
    parser.add_argument('--scaler', type=str, default='standard',
                        help='Scaler for the features. Options: "standard", "maxmin", "robust".')
    args = parser.parse_args()
    
    classifier = SVMClassifier(args)
    
    classifier.run()