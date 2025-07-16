import os
import warnings

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

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
        if self.df is None:
            raise ValueError("No data loaded! Please run load_data() first.")

        print("🧹 Starting data cleaning...")
        print(f"📊 Original shape: {self.df.shape}")
        print(f"📊 Missing values by column:")
        missing_info = self.df.isnull().sum()
        for col, count in missing_info[missing_info > 0].items():
            print(f"   {col}: {count} missing ({count / len(self.df) * 100:.1f}%)")

        df_cleaned = self.df.copy()

        columns_to_drop = ['source_url']
        existing_cols_to_drop = [col for col in columns_to_drop if col in df_cleaned.columns]
        if existing_cols_to_drop:
            df_cleaned.drop(columns=existing_cols_to_drop, inplace=True)
            print(f"🗑️  Dropped irrelevant columns: {existing_cols_to_drop}")

        critical_columns = ['model', 'range_km']
        existing_critical = [col for col in critical_columns if col in df_cleaned.columns]
        if existing_critical:
            initial_rows = len(df_cleaned)
            df_cleaned = df_cleaned.dropna(subset=existing_critical)
            dropped_rows = initial_rows - len(df_cleaned)
            if dropped_rows > 0:
                print(f"🗑️  Dropped {dropped_rows} rows missing critical data ({existing_critical})")

        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'range_km']

        for col in numeric_columns:
            if df_cleaned[col].isnull().any():
                missing_count = df_cleaned[col].isnull().sum()
                median_val = df_cleaned[col].median()
                df_cleaned[col].fillna(median_val, inplace=True)
                print(f"🔢 Filled {missing_count} missing values in '{col}' with median: {median_val:.2f}")

        categorical_columns = df_cleaned.select_dtypes(include=['object']).columns

        for col in categorical_columns:
            if df_cleaned[col].isnull().any():
                missing_count = df_cleaned[col].isnull().sum()

                if not df_cleaned[col].mode().empty:
                    mode_val = df_cleaned[col].mode()[0]
                    df_cleaned[col].fillna(mode_val, inplace=True)
                    print(f"📝 Filled {missing_count} missing values in '{col}' with mode: '{mode_val}'")
                else:
                    df_cleaned[col].fillna('Unknown', inplace=True)
                    print(f"📝 Filled {missing_count} missing values in '{col}' with 'Unknown'")

        df_cleaned.reset_index(drop=True, inplace=True)

        remaining_nan = df_cleaned.isnull().sum().sum()
        if remaining_nan > 0:
            print(f"⚠️  Warning: {remaining_nan} NaN values still remain!")
            print("Columns with remaining NaN:")
            remaining_cols = df_cleaned.isnull().sum()
            for col, count in remaining_cols[remaining_cols > 0].items():
                print(f"   {col}: {count}")
        else:
            print("✅ All missing values handled successfully!")

        print(f"📊 Final shape: {df_cleaned.shape}")
        print(f"📊 Rows removed: {self.df.shape[0] - df_cleaned.shape[0]}")

        self.df = df_cleaned

        return self.df

    def compare_models(self, target_column='range_km', test_size=0.2):
        """
        Compare all models and return performance metrics.

        Args:
            target_column (str): Name of the column to predict (default: 'range_km')
            test_size (float): Proportion of data to use for testing (default: 0.2)

        Returns:
            pd.DataFrame: Results with model names and their performance metrics
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        from sklearn.preprocessing import LabelEncoder
        import time

        if self.df is None:
            raise ValueError("No data loaded! Please run load_data() first.")

        print("🔧 Preparing data for model comparison...")
        print(f"📋 Available columns: {list(self.df.columns)}")

        # Check if the target column exists
        if target_column not in self.df.columns:
            print(f"❌ Target column '{target_column}' not found!")
            print(f"💡 Available columns: {list(self.df.columns)}")
            raise KeyError(f"Target column '{target_column}' not found in dataset")

        # Create a copy to avoid modifying the original data
        df_model = self.df.copy()

        # Better data cleaning - handle missing values more thoroughly
        print(f"📊 Missing values before cleaning: {df_model.isnull().sum().sum()}")

        # Drop rows with missing target values
        df_model = df_model.dropna(subset=[target_column])

        # Handle categorical variables (encode them as numbers)
        categorical_columns = df_model.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != target_column]

        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            # Fill missing categorical values with 'Unknown' before encoding
            df_model[col] = df_model[col].fillna('Unknown')
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            label_encoders[col] = le

        # Fill missing numeric values with median
        numeric_columns = df_model.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != target_column]

        for col in numeric_columns:
            if df_model[col].isnull().any():
                median_val = df_model[col].median()
                df_model[col] = df_model[col].fillna(median_val)
                print(f"   📝 Filled {col} missing values with median: {median_val}")

        print(f"📊 Missing values after cleaning: {df_model.isnull().sum().sum()}")

        # Separate features (X) and target (y)
        X = df_model.drop(columns=[target_column])
        y = df_model[target_column]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        print(f"📊 Training set size: {X_train.shape[0]}")
        print(f"📊 Test set size: {X_test.shape[0]}")
        print("\n🚀 Training models...")

        # Store results for each model
        results = []

        for model_name, model in self.models.items():
            print(f"   Training {model_name}...")

            # Time the training
            start_time = time.time()

            try:
                # Train the model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                training_time = time.time() - start_time

                # Store results
                results.append({
                    'Model': model_name,
                    'RMSE': round(rmse, 2),
                    'MAE': round(mae, 2),
                    'R²': round(r2, 4),
                    'Training Time (s)': round(training_time, 3)
                })

                print(f"   ✅ {model_name} - R²: {r2:.4f}")

            except Exception as e:
                print(f"   ❌ {model_name} failed: {e}")
                results.append({
                    'Model': model_name,
                    'RMSE': 'Failed',
                    'MAE': 'Failed',
                    'R²': 'Failed',
                    'Training Time (s)': 'Failed'
                })

        # Convert results to DataFrame and sort by R² score
        results_df = pd.DataFrame(results)

        # Sort by R² score (higher is better), handling failed models
        results_df_sorted = results_df.copy()
        results_df_sorted['R²_numeric'] = pd.to_numeric(results_df_sorted['R²'], errors='coerce')
        results_df_sorted = results_df_sorted.sort_values('R²_numeric', ascending=False)

        # Move failed models to the end
        failed_mask = results_df_sorted['R²_numeric'].isna()
        success_df = results_df_sorted[~failed_mask]
        failed_df = results_df_sorted[failed_mask]
        results_df_sorted = pd.concat([success_df, failed_df], ignore_index=True)

        results_df_sorted = results_df_sorted.drop('R²_numeric', axis=1)

        print("\n🏆 Model Comparison Results:")
        print("=" * 60)
        print(results_df_sorted.to_string(index=False))

        # Store results as instance variable for later use
        self.model_results = results_df_sorted
        self.label_encoders = label_encoders  # Save for future predictions

        return results_df_sorted


pipeline = EVPipeline()

# Load and clean your data
pipeline.load_data()
pipeline.clean_data()

# Compare all models
results = pipeline.compare_models()

# The results are also stored in pipeline.model_results for later use
print(f"\nBest model: {results.iloc[0]['Model']}")
print(f"Best R² score: {results.iloc[0]['R²']}")
