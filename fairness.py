import numpy as np
from argparse import ArgumentParser
from pandas import crosstab, read_csv
from scipy.stats import chi2_contingency
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


def test_independence(feature, label, threshold=0.05):
    """
    Tests the independence between a random variables distributed in `feature` 
    and `label` using the chi-square test. If the test p-value is less than the 
    specified `threshold`, there is evidence of dependence between the 
    variables. Otherwise, there is no evidence of dependence.

    Parameters:
    ---
    - `df_feature`: an array contains the categorical feature.
    - `df_label`: an array contains the label.
    - `p_value`: threshold for significance level, default is 0.05.

    Returns:
    ---
    A tuple of (`chi2_stat`, `p_val`) of the test.

    Example:
    ---
    >>> test_independence(df['Gender'], df['Approved'], threshold=0.01)
    """

    contingency_table = crosstab(feature, label)
    chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)

    print(f"[>] chi-square statistic: {chi2_stat:.4f}")
    print(f"[p] p-value: {p_val}")
    if p_val < threshold:
        print("[!] there is evidence of dependence between variables")
    else:
        print("[x] there is no evidence of dependence between variables")

    return (chi2_stat, p_val)


def compare_confusion_matrices(feature, y_pred, y_true):
    matrices = {}

    for group in np.unique(feature):
        mask = (feature == group)
        group_y_pred = y_pred[mask]
        group_y_test = y_true[mask]

        cfs_mat = confusion_matrix(group_y_test, group_y_pred)
        tn, fp, fn, tp = cfs_mat.ravel()

        print(f"{group}")
        print(f"| TP | FP |   | {tp:2d} | {fp:2d} |")
        print(f"|----|----| = |----|----|")
        print(f"| FN | TN |   | {fn:2d} | {tn:2d} |")
        print()

        matrices[group] = cfs_mat
    
    return matrices


def compare_approval_rate(feature, y_pred, y_true):
    """
    Compare the approval rate for different groups defined by the categorical 
    `feature` vector, given the model's predictions `y_pred` and the true value
    `y_true`.

    The approval rate is the probability for which the model thinks a certain 
    group from the `feature` should get their profile approved for credit card.
    It is calculated by (TP + FP)/(TP + FP + TN + FN).

    Parameters:
    ---
    - `feature`: array-like object containing the categorical feature values.
    - `y_pred`: array-like object containing the predicted labels.
    - `y_true`: array-like object containing the true labels.

    Returns:
    ---
    A dictionary containing the approval rate for each group.

    Example:
    ---
    ```
    >>> svm = SVC(...)
    >>> y_pred = svm.predict(x_test)
    >>> ethnicity = test_data['Ethnicity'].to_numpy()
    >>> compare_approval_rate(ethnicity, y_pred, y_true)
    ```
    """

    rate = {}

    for group in np.unique(feature):
        mask = (feature == group)
        group_y_pred = y_pred[mask]
        group_y_test = y_true[mask]

        cfs_mat = confusion_matrix(group_y_test, group_y_pred)
        tn, fp, fn, tp = cfs_mat.ravel()

        rate[group] = (tp + fp) / (tp + fp + tn + fn)

    for group, rate in rate.items():
        print(f"Group {group} \t- Approval rate: {rate:.2f}")

    return rate


def compare_demographic_parity(feature, y_pred, y_true):
    """
    Compare the demographic parity for different groups defined by the categorical 
    `feature` vector, given the model's predictions `y_pred` and the true value
    `y_true`.

    The demographic parity is the accuracy of the model in deciding if a certain 
    group from the `feature` should get their profile approved for credit card.
    It is calculated by (TP + TN)/(TP + FP + TN + FN).

    Parameters:
    ---
    - `feature`: array-like object containing the categorical feature values.
    - `y_pred`: array-like object containing the predicted labels.
    - `y_true`: array-like object containing the true labels.

    Returns:
    ---
    A dictionary containing the demographic parity for each group.

    Example:
    ---
    ```
    >>> svm = SVC(...)
    >>> y_pred = svm.predict(x_test)
    >>> ethnicity = test_data['Ethnicity'].to_numpy()
    >>> compare_demographic_parity(ethnicity, y_pred, y_true)
    ```
    """
    
    accuracy = {}

    for group in np.unique(feature):
        mask = (feature == group)
        group_y_pred = y_pred[mask]
        group_y_test = y_true[mask]

        cfs_mat = confusion_matrix(group_y_test, group_y_pred)
        tn, fp, fn, tp = cfs_mat.ravel()

        accuracy[group] = (tp + tn) / (tp + tn + fp + fn)

    for group, acc in accuracy.items():
        print(f"Group {group} \t- Accuracy: {acc:.2f}")

    return accuracy


def compare_equal_opportunity(feature, y_pred, y_true):
    """
    Compare the equal opportunity for different groups defined by the categorical 
    `feature` vector, given the model's predictions `y_pred` and the true value
    `y_true`.

    The equal opportunity is the true positive rate of the model in predicting 
    if a certain group from the `feature` should get their profile approved for 
    credit card. It is calculated by TP/(TP + FN).

    Parameters:
    ---
    - `feature`: array-like object containing the categorical feature values.
    - `y_pred`: array-like object containing the predicted labels.
    - `y_true`: array-like object containing the true labels.

    Returns:
    ---
    A dictionary containing the equal opportunity for each group.

    Example:
    ---
    ```
    >>> svm = SVC(...)
    >>> y_pred = svm.predict(x_test)
    >>> ethnicity = test_data['Ethnicity'].to_numpy()
    >>> compare_equal_opportunity(ethnicity, y_pred, y_true)
    ```
    """

    tpr = {}

    for group in np.unique(feature):
        mask = (feature == group)
        group_y_pred = y_pred[mask]
        group_y_test = y_true[mask]

        cfs_mat = confusion_matrix(group_y_test, group_y_pred)
        tn, fp, fn, tp = cfs_mat.ravel()

        tpr[group] = tp / (tp + fn)

    for group, tpr in tpr.items():
        print(f"Group {group} \t- TPR: {tpr:.2f}")

    return tpr


if __name__ == "__main__":
    parser = ArgumentParser(description='Fairness evaluation')
    parser.add_argument(
        '--feature', type=str, default='Ethnicity',
        help='named feature (column name) for which fairness will be evaluated. (Options: "Gender", "Age", "Debt", "Married", "BankCustomer", "Industry", "Ethnicity", "YearsEmployed", "PriorDefault", "Employed", "CreditScore", "DriversLicense", "Citizen", "Income")')
    
    args = parser.parse_args()
    col = args.feature

    # Read the train dataset
    df_train = read_csv("output/train_set.csv")
    print(f"\n-- Summary of feature")
    print(df_train[col].info())

    feature = df_train[col].to_numpy()
    label = df_train.iloc[:, -1].to_numpy()

    print(f"\n-- Chi-squared independence test")
    test_independence(feature, label)

    # Read the test dataset
    df_test = read_csv("output/test_set.csv")
    feature = df_test[col].to_numpy()
    y_true = df_test.iloc[:, -1].to_numpy()
    y_pred = read_csv("output/y_pred.csv")

    print(f"\n-- Compare confusion matrices")
    compare_confusion_matrices(feature, y_pred, y_true)

    print(f"\n-- Compare approval rate")
    compare_approval_rate(feature, y_pred, y_true)

    print(f"\n-- Compare demographic parity")
    compare_demographic_parity(feature, y_pred, y_true)

    print(f"\n-- Compare equal opportunity")
    compare_equal_opportunity(feature, y_pred, y_true)
