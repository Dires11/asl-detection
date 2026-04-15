from sklearn.utils import shuffle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np


class Data Preprocessor:
    def __init__(self):
        


def load_data():
    file_path_1 = 'revised_asl_sara.csv' # Replace with your actual file path
    file_path_2 = 'revised_asl_davit.csv'
    file_path_3 = 'revised_asl_arthur.csv'

    df1 = pd.read_csv(file_path_1)
    df2 = pd.read_csv(file_path_2)
    df3 = pd.read_csv(file_path_3)

    df = pd.concat([df1, df2, df3], ignore_index=True)

    df = shuffle(df, random_state=42)
    df = shuffle(df, random_state=42)
    print(len(df))

    return df

def split_data(df):
    y = df['label']
    X = df.drop('label', axis=1)



    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    AUGMENT_COPIES = 4      # each sample gets 4 noisy copies
    JITTER_STD     = 0.015  # ~1.5% of the normalised scale

    '''
    rng   = np.random.default_rng(42)
    noise = rng.normal(0, JITTER_STD, (len(X_train) * AUGMENT_COPIES, X_train.shape[1]))
    X_train= np.vstack([X_train, np.tile(X_train, (AUGMENT_COPIES, 1)) + noise])
    y_train = np.concatenate([y_train, np.tile(y_train, AUGMENT_COPIES)])
    '''
    return X_train, x_test, y_train, y_test

def train_model(X_train,y_train):
    rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",)   

    rf.fit(X_train, y_train)
    joblib.dump(rf, "rf_model.pkl")
    return rf

def test_model(rf,x_test):
    y_pred = rf.predict(x_test)
    return y_pred

def eval(y_test,y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", classification_rep)

if __name__ == "__main__":
    df = load_data()
    X_train,X_test,y_train, y_test = split_data(df)
    model = train_model(X_train,y_train)
    y_pred = test_model(model, X_test)
    eval(y_pred,y_test)
