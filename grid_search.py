# Validation of training + test selection of parameters, after feature selection, for paper:
#  "Evolutionary Optimization for the Classification of Small Molecules Regulating the 
#  Circadian Rhythm Period: A Reliable Assessment. Submmited to MDPI Algorithms 2025, with
#  preprint in Arxiv: https://arxiv.org/abs/2505.05485
import ast
import math
import time
import statistics
import sys

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn import datasets

def extract_target(dataset_name):
    """ Extract target from filename given as name_TARGET[._]whatever"""
    target = None
    first_underscore_pos = dataset_name.find('_')
    second_underscore_pos = dataset_name.find('_', first_underscore_pos + 1)
    dot_pos = dataset_name.find('.')

    if first_underscore_pos > 0:
        target = dataset_name[first_underscore_pos + 1:second_underscore_pos
        if second_underscore_pos > 0 else dot_pos]
    return target

def load_data(dataset_name):
    if dataset_name[-4:] == '.csv':
        data = pd.read_csv(dataset_name)
        target = extract_target(dataset_name)
        features = list(data.columns)
        if target not in features:
            print(f'Target "{target}" not present in data')
            return None, None
        features.remove(target)
        X = data[features].values
        y = data[target].values
        print(f'Classes: {np.unique(y)}')
    else:
        raise Exception('Unsupported file format')

    # Convert class to numeric for xgboost classifier
    if not isinstance(y[0], (int, float)):
        le = sklearn.preprocessing.LabelEncoder()
        y = le.fit_transform(y)

    return X, y, features

def main(args):
    # Load data set
    dataset_name = args.dataset_name[0]
    print(f'Dataset: {dataset_name}')
    X, y, all_features = load_data(dataset_name)
    if X is None or y is None:
        print('Invalid data set')
        exit(1)

    # Selected features
    if args.features:
        selected_features = ast.literal_eval(args.features)
        if isinstance(selected_features, list) and all(isinstance(col, bool) for col in selected_features):
            print(f"Selected features: {list(np.array(all_features)[selected_features])}")
        elif isinstance(selected_features, list):
            print(f"Selected features: {selected_features}")
            selected_features = [v in selected_features for v in all_features]
        else:
            raise ValueError("Decoded features must be a list of strings with feature names or a list of N booleans.")
        X = X[:,selected_features]

    n_features = X.shape[1]

    # Classifier parameters
    if args.classifier == 'knn':
        classifier = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [1,3,5,7,9],
        }
    elif args.classifier == 'rf':
        classifier = RandomForestClassifier()
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 1, 2, 3, 4, 5, 6],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [1, 3, 5],
            'max_features': list(range(1, int(math.sqrt(n_features))+1, 2)) #[1 ,2 ,3 ,4 ,5,'sqrt(num_features)'],
        }
    elif args.classifier == 'dtc':
        classifier = DecisionTreeClassifier()
        param_grid = {
            'max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
            'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'max_features': list(range(1, int((n_features)) + 1, 2))  # [1 ,2 ,3 ,4 ,5,'sqrt(num_features)'],
        }
    elif args.classifier == 'etc':
        classifier = ExtraTreesClassifier()
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 1, 2, 3, 4, 5, 6],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [1, 3, 5],
            'max_features': list(range(1, int(math.sqrt(n_features)) + 1, 2))  # [1 ,2 ,3 ,4 ,5,'sqrt(num_features)'],
        }
    elif args.classifier == 'xgbc':
        from xgboost import XGBClassifier
        classifier = XGBClassifier()
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7, 10],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.5, 0.7],
            'colsample_bytree': [0.5, 0.7],
        }
    else:
        raise Exception('Unsupported classifier to optimize')


    # Experimentation
    avg_validation_accuracies = []
    avg_validation_f1_rep = []
    avg_validation_precision_rep = []
    avg_validation_recall_rep = []

    for ext_repeat_i in range(10):
        print(f'\n--- Repetición {ext_repeat_i} ---')
        ext_partitioner = StratifiedKFold(args.valfolds, shuffle=True)
        ext_cv_partitions = []
        for train_indexes, test_indexes in ext_partitioner.split(X, y):
            X_work, X_validation = X[train_indexes], X[test_indexes]
            y_work, y_validation = y[train_indexes], y[test_indexes]
            ext_cv_partitions.append( (X_work, X_validation, y_work, y_validation) )

        val_accuracies = []
        val_f1 = []
        val_precision = []
        val_recall = []

        for ext_cv_i in range( len(ext_cv_partitions) ):
            X_work, X_validation, y_work, y_validation = ext_cv_partitions[ext_cv_i]

            start_time = time.time()

            grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_work, y_work)

            total_time = time.time() - start_time

            # Best model found
            avg_test_accuracy = grid_search.cv_results_['mean_test_score'][grid_search.best_index_]
            std_test_accuracy = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
            print("Hiperparámetros:", grid_search.best_params_, "T: {:.2f}s".format(total_time))
            print(f"Test accuracy:  {avg_test_accuracy}   std: {std_test_accuracy}")
            predictor = grid_search.best_estimator_ 

            # Evaluate
            y_pred = predictor.predict(X_validation)
            val_accuracies.append(accuracy_score(y_validation, y_pred) * y_validation.shape[0])
            val_f1.append(        f1_score(y_validation, y_pred, average='weighted') * y_validation.shape[0])
            val_precision.append( precision_score(y_validation, y_pred, average='weighted') * y_validation.shape[0])
            val_recall.append(    recall_score(y_validation, y_pred, average='weighted') * y_validation.shape[0])

            print(f"Validación {ext_cv_i}: {val_accuracies[-1]}")
            print("F1 Score:       {:.4f}".format(val_f1[-1]))
            print("Precision:      {:.4f}".format(val_precision[-1]))
            print("Recall:         {:.4f}\n".format(val_recall[-1]))

            sys.stdout.flush()

        avg_validation_accuracy = sum(val_accuracies) / y.shape[0]
        print(f"Average accuracy: {avg_validation_accuracy}")
        avg_validation_accuracies.append(avg_validation_accuracy)

        avg_validation_f1 = sum(val_f1) / y.shape[0]
        print(f"Average f1: {avg_validation_f1}")
        avg_validation_f1_rep.append(avg_validation_f1)

        avg_validation_precision = sum(val_precision) / y.shape[0]
        print(f"Average precision: {avg_validation_precision}")
        avg_validation_precision_rep.append(avg_validation_precision)

        avg_validation_recall = sum(val_recall) / y.shape[0]
        print(f"Average recall: {avg_validation_recall}")
        avg_validation_recall_rep.append(avg_validation_accuracy)

    print(avg_validation_accuracies)
    print(f"Media repeticiones Accuracy:{statistics.mean(avg_validation_accuracies)}")
    print(avg_validation_f1_rep)
    print(f"Media repeticiones F1:{statistics.mean(avg_validation_f1_rep)}")
    print(avg_validation_precision_rep)
    print(f"Media repeticiones Precision:{statistics.mean(avg_validation_precision_rep)}")
    print(avg_validation_recall_rep)
    print(f"Media repeticiones Recall:{statistics.mean(avg_validation_recall_rep)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run grid optimization for a classifier',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset_name', nargs=1, help='Data set to learn from')
    parser.add_argument('-f', '--features', type=argparse.FileType('r'), default=None, help="Path to the selected features file that may contain a python list of strings with feature names or a python list of boolean values indicating if feature at its position is selected")
    parser.add_argument('--valfolds',       default=100,  help='External validation number of CV folds', type=int)
    parser.add_argument('-c', '--classifier', type=str, default='rf', help='classifier to optimize: knn, rf, dtc, etc, xgbc')
    args = parser.parse_args()

    if args.features:
        features_content = args.features.read()
        args.features.close()
        args.features = features_content

    main(args)