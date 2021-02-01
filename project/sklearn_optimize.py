import optuna
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from imblearn.over_sampling import SMOTE
from utils.data_utils import preprocess

def load_data():
    mirage_csv_file = "C:/CS-GO-Grenade-Classification/project/data/train-grenades-de_mirage.csv"
    inferno_csv_file = "C:/CS-GO-Grenade-Classification/project/data/train-grenades-de_inferno.csv"
    inferno = pd.read_csv(mirage_csv_file, index_col = 0)
    mirage = pd.read_csv(inferno_csv_file, index_col = 0)

    raw_data = pd.concat([inferno, mirage])
    X, y = preprocess(raw_data)
    return X, y

class Objective():

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.classifiers = ["LGBM", "XGB", "GaussianNB", "LogisticRegression", "ET"]
        self.class_weight = "balanced"

    def __call__(self, trial):
        X = np.copy(self.X)
        y = np.copy(self.y)
        params = {}

        clf_name = trial.suggest_categorical("clf_name", self.classifiers)
        if clf_name == "LGBM":
            params["boosting_type"] = trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"])
            params["num_leaves"] = trial.suggest_int("num_leaves", 10, 100)
            params["max_depth"] = trial.suggest_int("max_depth", 2, 50)
            params["learning_rate"] = trial.suggest_float("learning_rate", 1e-6, 1, log = True)
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 500)
            clf = LGBMClassifier(is_unbalance = True, **params)
        elif clf_name == "XGB":
            params["booster"] = trial.suggest_categorical("booster", ["gbtree", "dart", "gblinear"])
            params["learning_rate"] = trial.suggest_float("learning_rate", 1e-6, 1, log = True)
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 500)
            params["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-10, 1e2)
            params["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-10, 1e2)
            clf = XGBClassifier(n_jobs = -1, **params)
        elif clf_name == "GaussianNB":
            params["var_smoothing"] = trial.suggest_float("var_smoothing", 1e-10, 1e-2)
            clf = GaussianNB(**params)
        elif clf_name == "LogisticRegression":
            params["tol"] = trial.suggest_float("tol", 1e-5, 1e-2)
            params["C"] = trial.suggest_float("C", 1e-10, 1e2)
            params["fit_intercept"] = trial.suggest_categorical("fit_intercept", [True, False])
            params["max_iter"] = trial.suggest_int("max_iter", 100, 1000)
            clf = LogisticRegression(n_jobs = -1, class_weight = self.class_weight, **params)
        else:
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 500)
            params["max_depth"] = trial.suggest_int("max_depth", 2, 50)
            params["criterion"] = trial.suggest_categorical("criterion", ["gini", "entropy"])
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 50)
            clf = ExtraTreesClassifier(n_jobs = -1, class_weight = self.class_weight, **params)

        scores = cross_val_score(clf, X, y, scoring = "balanced_accuracy")
        return scores.mean()

if __name__ == "__main__":

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)
    objective = Objective(X_train, y_train)
    
    study = optuna.create_study(direction = "maximize")
    study.optimize(objective, n_trials = 200)
    trial = study.best_trial
    print(f"Best trial: {trial.value}")
    print("Best trial params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")





