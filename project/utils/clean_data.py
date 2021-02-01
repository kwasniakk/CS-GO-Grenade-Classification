import pandas as pd

COLUMNS_TO_DROP = ["demo_id", "demo_round_id", "weapon_fire_id", "round_start_tick"]
DUMMY_COLS = ["LABEL", "team", "TYPE", "map_name"]


def preprocess(df):
    data_dropped =  df.drop(columns = COLUMNS_TO_DROP)
    data_cleaned = pd.get_dummies(data_dropped, columns = DUMMY_COLS, drop_first = True)
    X = data_cleaned.drop(columns = ["LABEL_True"])
    y = data_cleaned["LABEL_True"]
    return X, y

