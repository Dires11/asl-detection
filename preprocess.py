import pandas as pd
import numpy as np

class DataHandling():
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        return self.data


    def drop_column(self):
        cols_to_drop = [f"z_{i}" for i in range(21)]
        self.data.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    
    def clean_data(self):
        self.data = self.data.dropna()
        self.data = self.data.drop_duplicates()

    def scale_features(self, feature_cols):
        self.data[feature_cols] = self.scaler.fit_transform[self.data[feature_cols]]
    
    
    def pos_invariance(self):
        for i in range(21):
            self.data[f"x_{i}"] = self.data["x_0"]-self.data[f"x_{i}"]
            self.data[f"y_{i}"] = self.data["y_0"]-self.data[f"y_{i}"]
        return self.data.head()
    
    def scale_invariance(self):
        
        for i in range(21):
            if i != 0:
                self.data[f"x_{i}"] = self.data[f"x_{i}"]/self.data["x_12"]
                self.data[f"y_{i}"] = self.data[f"y_{i}"]/self.data["y_12"]
        return self.data.head()
    
    def save_df(self, name):
        self.data.to_csv(f"revised_asl_{name}.csv", index = False)
        return self.data


def apply_data_handling(file,name):
    data = DataHandling(file)
    data.load_data()
    data.drop_column()
    data.pos_invariance()
    data.scale_invariance()
    data.save_df(name)


if __name__ == "__main__":
    asl_sara = apply_data_handling("asl_landmarks_sara.csv", "sara")
    asl_davit = apply_data_handling("asl_landmarks_davit.csv", "davit")
    asl_arthur = apply_data_handling("asl_landmarks_arthur.csv", "arthur")


