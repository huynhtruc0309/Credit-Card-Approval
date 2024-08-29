import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

class Preprocessor():
    def __init__(self):
        super(Preprocessor, self).__init__()

    @staticmethod
    def rename_columns(df, col_names):
        """
        This function renames the columns of a pandas DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame whose columns are to be renamed.
            col_names (list): A list of new column names. The length and order should match the columns of the DataFrame.

        Returns:
            pandas.DataFrame: The DataFrame with renamed columns.
        """
        df.columns = col_names
        return df
    
    @staticmethod
    def calculate_missing_values(df):
        """
        This function calculates the number and percentage of missing values in each column of a DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame for which missing values are to be calculated.

        Returns:
            pandas.DataFrame: A DataFrame with each column's name, the number of missing values in that column, 
                            and the percentage of total values in the column that are missing.
        """
        missing_values = []
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_percentage = np.round(100 * missing_count / len(df), 2)
            missing_values.append({'Feature': col, 'NumberMissing': missing_count, 'PercentageMissing': missing_percentage})
        mv_df = pd.DataFrame(missing_values)
        return mv_df
    
    @staticmethod
    def replace_missing_values(df):
        """
        This function replaces missing values, represented as '?', in a DataFrame with NaN.

        Args:
            df (pandas.DataFrame): The DataFrame in which missing values are to be replaced.

        Returns:
            pandas.DataFrame: The DataFrame with '?' replaced by NaN.
        """
        df = df.replace('?', np.nan)
        return df
        
    @staticmethod
    def get_most_correlated(df, continuous_variables):
        """
        This function calculates the correlation matrix for the continuous variables in the DataFrame and 
        identifies the pair of variables that have the highest correlation.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data.
            continuous_variables (list): A list of column names representing the continuous variables.

        Returns:
            tuple: A pair of column names that have the highest correlation.
        """
        corr_matrix = df[continuous_variables].corr()
        corr_matrix_np = corr_matrix.values
        np.fill_diagonal(corr_matrix_np, 0)
        most_correlated_idx = np.unravel_index(np.argmax(corr_matrix_np, axis=None), corr_matrix_np.shape)
        most_correlated_features = corr_matrix.columns[list(most_correlated_idx)]
        return most_correlated_features[0], most_correlated_features[1]
    
    @staticmethod
    def init_scaler(scaler = 'standard'):
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
    def label_encode_categorical(data, target_feature='Approved'):
        """
        Converts categorical data into numerical form using LabelEncoder. 
        This is necessary because machine learning algorithms typically work with numerical data.

        Args:
            data (pd.DataFrame): The data to be encoded.

        Returns:
            pd.DataFrame: The data with categorical variables encoded as numerical values.
        """
        le = LabelEncoder()
        for col in data.columns:
            if col == target_feature:
                data.loc[:, col] = data.loc[:, col].map({'-': 0, '+': 1})
            
            if data[col].dtypes=='object':
                data.loc[:, col] = le.fit_transform(data.loc[:, col])
        return data
    
    @staticmethod
    def fill_nan_by_imputer(data):
        """
        This function replaces missing values in the DataFrame using an imputer. 
        The imputer replaces missing values with the mean value of the respective column.

        Args:
            data (pandas.DataFrame): The DataFrame in which missing values are to be replaced.

        Returns:
            pandas.DataFrame: The DataFrame with missing values replaced by the mean of the respective column.
        """
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        return pd.DataFrame(imp.fit_transform(data), columns = data.columns)
        
    
    def handle_continuous_values(self, df, continuous_variables):
        """
        This function handles missing values in continuous variables of a DataFrame. 
        It first identifies the two most correlated continuous variables. 
        If both have missing values, it uses an imputer to fill them. 
        If only one has missing values, it uses a linear regression model to predict and fill them.

        Args:
            df (pandas.DataFrame): The DataFrame in which missing values are to be handled.
            continuous_variables (list): A list of column names representing the continuous variables.

        Returns:
            pandas.DataFrame: The DataFrame with missing values in continuous variables handled.
        """
        feature1, feature2 = self.get_most_correlated(df, continuous_variables)

        nan_f1_count = df[feature1].isna().sum()
        nan_f2_count = df[feature2].isna().sum()
        
        if nan_f1_count == 0 and nan_f2_count == 0:
            return df
        elif nan_f1_count > 0 and nan_f2_count > 0:
            new_df = self.fill_nan_by_imputer(data[continuous_variables])
            return df.update(new_df)
        
        miss_val_feature, considered_feature = pd.DataFrame(), pd.DataFrame()
        if nan_f1_count > 0 and nan_f2_count == 0:
            miss_val_feature, considered_feature = feature1, feature2
        else:
            miss_val_feature, considered_feature = feature2, feature1
        
        data = df.dropna(subset=[miss_val_feature, considered_feature])
        X_train = data[[considered_feature]]
        y_train = data[miss_val_feature]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        missing_data = df[df[miss_val_feature].isna() & df[considered_feature].notna()]
        X_test = missing_data[[considered_feature]]
        predicted_age = model.predict(X_test)
        
        df.loc[missing_data.index, miss_val_feature] = predicted_age
        
        # nan_count = df[miss_val_feature].isna().sum()
        # print(f"Number of NaN values in the {miss_val_feature} column after filling:", nan_count)

        df[miss_val_feature] = df[miss_val_feature].astype(float)
        
        # print("Intercept:", model.intercept_)
        # print(f"Coefficient for {considered_feature}", model.coef_[0])
        
        for col in continuous_variables:
            df[col] = np.log(df[col] + 1)
        
        return df
        
    def handle_categorical_variables(self, df, categorical_variables, feature_names):
        """
        This function handles missing values in categorical variables of a DataFrame. 
        For binary categorical variables, it uses label encoding. 
        For non-binary categorical variables, it uses a Decision Tree Classifier to predict and fill missing values.

        Args:
            df (pandas.DataFrame): The DataFrame in which missing values are to be handled.
            categorical_variables (list): A list of column names representing the categorical variables.
            feature_names (list): A list of all non-missing data feature names in the DataFrame.

        Returns:
            pandas.DataFrame: The DataFrame with missing values in categorical variables handled.
        """
        for col in categorical_variables:
            if df[col].isna().sum() > 0 and df[col].nunique() == 2:
                df[col] = self.label_encode_categorical(df[[col]])
            elif df[col].isna().sum() > 0 and df[col].nunique() > 2:
                if col == 'Married':
                    df['Married'].replace('l', 'y', inplace=True)
                elif col == 'BankCustomer':
                    df['BankCustomer'].replace('gg', 'g', inplace=True)
                elif col == 'Ethnicity':
                    df.loc[df['Ethnicity'].isin(['j','z','dd','n','o']),'Ethnicity']='Other'
                selected_columns = feature_names.tolist()[:-1] + [col]
                missing_rows = df[selected_columns][df[selected_columns].isna().any(axis=1)]
                non_missing_rows = df[selected_columns].dropna(subset=[col])
                
                le = LabelEncoder()
                non_missing_rows[col] = abs(1 - le.fit_transform(non_missing_rows[col]))

                X_train = non_missing_rows.drop(columns=[col])
                y_train = non_missing_rows[col]

                model = DecisionTreeClassifier()
                model.fit(X_train, y_train)

                missing_rows.loc[:, col] = model.predict(missing_rows.drop(columns=[col]))
                df[col] = pd.concat([missing_rows, non_missing_rows], axis=0)[col]
        return df
    
    @staticmethod
    def split_data(data, target_feature='Approved', test_size=0.25):
        """
        This function splits the input DataFrame into training and testing sets. 
        The target feature is removed from the DataFrame to form the feature set. 
        The target feature forms the label set. 

        Args:
            data (pandas.DataFrame): The DataFrame to be split.
            target_feature (str, optional): The name of the target feature. Defaults to 'Approved'.
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.

        Returns:
            tuple: The feature set for training, the feature set for testing, 
                the label set for training, and the label set for testing.
        """
        X = data.drop(target_feature, axis=1)
        y = data[target_feature]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def normalize_data(scaler, X_train, X_test):
        """
        This function normalizes the training and testing feature sets using the provided scaler. 
        The scaler is fitted on the training set and used to transform both the training and testing sets.

        Args:
            scaler (sklearn.preprocessing.StandardScaler or similar): The scaler to be used for normalization.
            X_train (pandas.DataFrame or numpy.ndarray): The training feature set.
            X_test (pandas.DataFrame or numpy.ndarray): The testing feature set.

        Returns:
            tuple: The normalized training and testing feature sets.
        """
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test
    
    @staticmethod
    def save_to_csv(data, output_dir='./data', file_name='preprocessed_credit_card_approvals.csv'):
        """
        This function saves the preprocessed DataFrame to a CSV file.

        Args:
            data (pandas.DataFrame): The DataFrame to be saved.
            output_dir (str, optional): The directory where the file will be saved. Defaults to './data/'.
            file_name (str, optional): The name of the file. Defaults to 'preprocessed_credit_card_approvals.csv'.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        data.to_csv(os.path.join(output_dir, file_name), index=False)

def run(args):    
    preprocessor = Preprocessor()
    
    df = pd.read_csv(args.data, header=None)
    
    col_names = ['Gender', 'Age', 'Debt', 'Married', 'BankCustomer', 'Industry', 'Ethnicity', 'YearsEmployed', 'PriorDefault', 'Employed', 'CreditScore', 'DriversLicense', 'Citizen', 'ZipCode', 'Income', 'Approved']
    df = preprocessor.rename_columns(df, col_names)
    df.drop('ZipCode', axis=1, inplace=True)
    
    df = preprocessor.replace_missing_values(df)
    
    continuous_variables = ['Age', 'Debt', 'YearsEmployed', 'CreditScore', 'Income']
    df = preprocessor.handle_continuous_values(df, continuous_variables)
    
    categorical_variables = ['Gender', 'Married', 'BankCustomer', 'Industry', 'Ethnicity', 'PriorDefault', 'Employed', 'DriversLicense', 'Citizen', 'Approved']
    mv = preprocessor.calculate_missing_values(df[categorical_variables])
    no_missing_data = mv[mv['NumberMissing'] == 0]
    no_missing_data_feature_names = no_missing_data['Feature'].values

    cleaned_df = df[no_missing_data_feature_names]
    cleaned_df = preprocessor.label_encode_categorical(cleaned_df, args.target)
    df.update(cleaned_df)

    df = preprocessor.handle_categorical_variables(df, categorical_variables, no_missing_data_feature_names)
    
    preprocessor.save_to_csv(df, args.output, 'preprocessed_credit_card_approvals.csv')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument('--data', type=str, default='./data/raw_credit_card_approvals.csv',
                        help='Path to the dataset.')
    parser.add_argument('--output', type=str, default='./data',
                        help='Path to the output directory.')
    parser.add_argument('--target', type=str, default='Approved',
                        help='Target value to classify.')
    parser.add_argument('--test_size', type=float, default=0.25,
                        help='Size of the test set.')
    parser.add_argument('--scaler', type=str, default='standard',
                        help='Scaler for the features. Options: "standard", "maxmin", "robust".')
    args = parser.parse_args()
    
    run(args)
    