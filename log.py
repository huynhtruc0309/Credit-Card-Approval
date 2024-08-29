import csv
import os

class Logger:
    def __init__(self, output_dir):
        self.performance = output_dir + '/model_performance.csv'
        self.clf_report = output_dir + '/classification_report.csv'
        
        if not os.path.exists(self.performance):
            with open(self.performance, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["run_id", "best_C", "best_gamma", "best_kernel", "accuracy", "TN", "FP", "FN", "TP"])
            self.run_id = 1
        else:
            with open(self.performance, 'r', newline='') as file:
                last_line = list(csv.reader(file))[-1]
                self.run_id = int(last_line[0]) + 1

        if not os.path.exists(self.clf_report):
            with open(self.clf_report, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["run_id", "label", "precision", "recall", "f1-score", "support"])

    def log(self, best_params, accuracy, conf_mat, clf_report):
        # Extract the best parameters
        best_C = best_params['C']
        best_gamma = best_params['gamma']
        best_kernel = best_params['kernel']

        # Extract the confusion matrix values
        TN, FP, FN, TP = conf_mat.ravel()

        with open(self.performance, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.run_id, best_C, best_gamma, best_kernel, accuracy, TN, FP, FN, TP])

        with open(self.clf_report, 'a', newline='') as file:
            writer = csv.writer(file)
            for class_label, row in clf_report.iterrows():
                writer.writerow([self.run_id, class_label] + row.tolist())
                
        self.run_id += 1