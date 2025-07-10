import os
import warnings

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import seaborn as sns
warnings.filterwarnings("ignore")

class EVPipeline:
    def __init__(self):
        self.file_path = None
        self.df = None

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=537),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=537),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=537),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=537),
        "Support Vector Regressor": SVR(),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5)
    }

    def load_data(self, csv_path=None):
        """Load EV data with pandas."""

        if csv_path is None:
            # Try to find the file automatically
            possible_paths = [
                'dataset/electric_vehicles_spec_2025.csv',
                'EvMLPipeline/dataset/electric_vehicles_spec_2025.csv',
                'electric_vehicles_spec_2025.csv'
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    csv_path = path
                    break

            if csv_path is None:
                raise FileNotFoundError("Could not find EV CSV file")

        try:
            print(f"📖 Reading CSV file: {csv_path}")

            # Read with pandas - basic approach
            self.df = pd.read_csv(csv_path)
            self.file_path = csv_path

            print(f"✅ Successfully loaded data!")
            print(f"   📊 Shape: {self.df.shape}")
            print(f"   📝 Columns: {list(self.df.columns)}")

            return self.df

        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            raise

    def clean_data(self):
        df_cleaned = self.df.dropna(subset=['model', 'cargo_volume_l'])

        # Fill numeric missing values with median
        for col in df_cleaned.select_dtypes(include=[np.number]).columns:
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)

        # Drop irrelevant columns
        df_cleaned.drop(columns=['source_url'], inplace=True)
        df_cleaned.reset_index(drop=True, inplace=True)

        df_cleaned.head()


