import pandas as pd
import pickle
import argparse
COLUMNS_TO_DROP = ["demo_id", "demo_round_id", "weapon_fire_id", "round_start_tick"]
DUMMY_COLS = ["team", "TYPE", "map_name"]


def load_data(csv_file):
    data = pd.read_csv(csv_file, index_col = 0)
    X = preprocess(data)
    return X

def preprocess(df):
    data_dropped =  df.drop(columns = COLUMNS_TO_DROP)
    data_cleaned = pd.get_dummies(data_dropped, columns = DUMMY_COLS, drop_first = True)
    return data_cleaned

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-csv_file", "--csv_file", type=str, default="test.csv")
    args = parser.parse_args()
    data = pd.read_csv(args.csv_file)
    X = preprocess(data)
    clf = pickle.load(open("saved_model.sav", "rb"))
    y_pred = clf.predict(X)
    data["RESULT"] = y_pred
    data.to_csv(args.csv_file)


