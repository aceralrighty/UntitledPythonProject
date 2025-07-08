import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from transformers import pipeline

warnings.filterwarnings('ignore')


class StockPredictionPipeline:
    def __init__(self, sample_size=10000, use_sentiment=False):
        self.dataset = None
        self.df = None
        self.sentiment_analyzer = None
        self.scaler = StandardScaler()
        self.model = None
        self.features = []
        self.sample_size = sample_size  # Limit dataset size for faster training
        self.use_sentiment = use_sentiment  # Skip sentiment analysis for speed

    def load_data(self):
        """Load the stock news dataset"""
        print("Loading dataset...")
        self.dataset = load_dataset("oliverwang15/us_stock_news_with_price")
        self.df = pd.DataFrame(self.dataset['train'])

        # Sample data for faster processing
        if len(self.df) > self.sample_size:
            print(f"Sampling {self.sample_size} rows from {len(self.df)} total rows")
            self.df = self.df.sample(n=self.sample_size, random_state=42)

        print(f"Dataset loaded with {len(self.df)} rows")
        return self.df

    def explore_data(self):
        """Explore the dataset structure"""
        print("\n=== Dataset Info ===")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print("\n=== Sample Data ===")
        print(self.df.head())

        print("\n=== Data Types ===")
        print(self.df.dtypes)

        print("\n=== Missing Values ===")
        print(self.df.isnull().sum())

        if 'date' in self.df.columns:
            print("\n=== Date Range ===")
            print(f"From: {self.df['date'].min()}")
            print(f"To: {self.df['date'].max()}")

    def preprocess_data(self):
        """Preprocess the data for ML"""
        print("\nPreprocessing data...")

        # Convert date column if it exists
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df = self.df.sort_values('date')

        # Handle missing values in basic columns
        basic_cols = ['date', 'stock', 'title', 'content', 'trading_date', 'exact_trading_date']
        for col in basic_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('unknown')

        # Extract features from text (news headlines/content)
        if 'title' in self.df.columns or 'content' in self.df.columns:
            self.extract_text_features()

        # Create target variable FIRST (before price features)
        self.create_target_variable()

        # Extract price features from historical data only
        self.extract_price_features()

        print(f"Data preprocessed. Shape: {self.df.shape}")
        print(f"Target variable stats: mean={self.df['target'].mean():.6f}, std={self.df['target'].std():.6f}")

    def extract_text_features(self):
        """Extract features from news text using sentiment analysis"""
        if not self.use_sentiment:
            print("Skipping sentiment analysis for faster processing")
            # Just add basic text features
            text_col = None
            for col in ['title', 'content', 'headline', 'news']:
                if col in self.df.columns:
                    text_col = col
                    break

            if text_col:
                self.df['text_length'] = self.df[text_col].str.len().fillna(0)
                self.df['text_word_count'] = self.df[text_col].str.split().str.len().fillna(0)
            return

        print("Extracting text features...")

        # Get text column
        text_col = None
        for col in ['title', 'content', 'headline', 'news']:
            if col in self.df.columns:
                text_col = col
                break

        if text_col is None:
            print("No text column found. Skipping text feature extraction.")
            return

        try:
            # Initialize sentiment analyzer
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",  # Faster model
                return_all_scores=True
            )

            # Calculate sentiment scores for sample
            sample_size = min(1000, len(self.df))  # Limit for speed
            sentiments = []

            for i, text in enumerate(self.df[text_col].head(sample_size)):
                try:
                    result = self.sentiment_analyzer(str(text)[:256])  # Shorter text
                    scores = {item['label']: item['score'] for item in result}
                    sentiments.append(scores)
                except:
                    sentiments.append({'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 1})

            # Add sentiment features
            sentiment_df = pd.DataFrame(sentiments)
            for col in sentiment_df.columns:
                self.df[f'sentiment_{col}'] = 0
                self.df[f'sentiment_{col}'].iloc[:sample_size] = sentiment_df[col]

            print(f"Text features extracted for {sample_size} samples")

        except Exception as e:
            print(f"Error in text feature extraction: {e}")

        # Basic text features
        self.df['text_length'] = self.df[text_col].str.len().fillna(0)
        self.df['text_word_count'] = self.df[text_col].str.split().str.len().fillna(0)

    def extract_price_features(self):
        """Extract technical indicators from price data"""
        print("Extracting price features...")

        # Find time series columns (ts_X format)
        ts_cols = [col for col in self.df.columns if col.startswith('ts_')]

        if ts_cols:
            # Sort ts columns by their numeric value
            ts_cols_sorted = sorted(ts_cols, key=lambda x: int(x.split('_')[1]))

            # Only use historical data (ts_0 and earlier) to avoid data leakage
            historical_cols = [col for col in ts_cols_sorted if int(col.split('_')[1]) <= 0]
            print(f"Using historical time series columns: {historical_cols}")

            # Calculate technical indicators using only historical data
            for col in historical_cols:
                if self.df[col].dtype in ['float64', 'int64']:
                    # Price differences (momentum)
                    if 'ts_0' in self.df.columns:
                        self.df[f'{col}_diff_from_ts0'] = self.df[col] - self.df['ts_0']

                    # Rolling statistics within historical window
                    hist_df = self.df[historical_cols].fillna(method='ffill')
                    if len(historical_cols) > 1:
                        self.df[f'{col}_historical_mean'] = hist_df.mean(axis=1)
                        self.df[f'{col}_historical_std'] = hist_df.std(axis=1)
                        self.df[f'{col}_historical_max'] = hist_df.max(axis=1)
                        self.df[f'{col}_historical_min'] = hist_df.min(axis=1)

            # Calculate price trends and volatility
            if len(historical_cols) > 5:
                price_matrix = self.df[historical_cols].values
                # Price slope (linear trend)
                x = np.arange(len(historical_cols))
                slopes = []
                for i in range(len(self.df)):
                    y = price_matrix[i]
                    if not np.isnan(y).all():
                        slope = np.polyfit(x, y, 1)[0]
                        slopes.append(slope)
                    else:
                        slopes.append(0)
                self.df['price_slope'] = slopes

                # Price volatility
                self.df['price_volatility'] = np.std(price_matrix, axis=1)

        print("Price features extracted")

    def create_target_variable(self):
        """Create target variable for prediction"""
        # Find time series columns (ts_X format)
        ts_cols = [col for col in self.df.columns if col.startswith('ts_')]

        if ts_cols:
            # Sort ts columns by their numeric value
            ts_cols_sorted = sorted(ts_cols, key=lambda x: int(x.split('_')[1]))
            print(f"Found time series columns: {ts_cols_sorted}")

            # Use ts_0 as the reference point (news publication time)
            if 'ts_0' in self.df.columns:
                reference_col = 'ts_0'
            else:
                reference_col = ts_cols_sorted[0]

            # Create target: predict price movement from ts_0 to ts_1, ts_2, etc.
            future_targets = ['ts_1', 'ts_2', 'ts_3', 'ts_5', 'ts_10', 'ts_15']

            for target_col in future_targets:
                if target_col in self.df.columns:
                    # Price change from reference time to future time
                    self.df[f'target_{target_col}'] = self.df[target_col] - self.df[reference_col]
                    self.df[f'target_{target_col}_pct'] = (self.df[target_col] - self.df[reference_col]) / self.df[
                        reference_col]
                    print(f"Created target {target_col}: mean={self.df[f'target_{target_col}'].mean():.6f}")

            # Use ts_1 as primary target (1 time step forward)
            if 'target_ts_1' in self.df.columns:
                self.df['target'] = self.df['target_ts_1']
                self.df['target_pct'] = self.df['target_ts_1_pct']
                print(f"Primary target created: mean={self.df['target'].mean():.6f}, std={self.df['target'].std():.6f}")
            else:
                print("No suitable future target found, creating dummy target")
                self.df['target'] = 0
        else:
            print("No time series columns found. Available columns:", list(self.df.columns))
            self.df['target'] = 0

    def prepare_features(self):
        """Prepare feature matrix for ML"""
        print("Preparing features...")

        # Check if target exists
        if 'target' not in self.df.columns:
            print("Target variable not found. Creating dummy target.")
            self.df['target'] = 0

        # Select feature columns - ONLY USE HISTORICAL DATA (ts_0 and earlier)
        exclude_cols = ['date', 'stock', 'title', 'content', 'trading_date', 'exact_trading_date',
                        'target', 'target_pct']

        # Exclude future time series data to prevent data leakage
        future_ts_cols = [col for col in self.df.columns if col.startswith('ts_') and int(col.split('_')[1]) > 0]
        exclude_cols.extend(future_ts_cols)

        # Also exclude any target columns we created
        target_cols = [col for col in self.df.columns if col.startswith('target_')]
        exclude_cols.extend(target_cols)

        print(f"Excluding future data columns: {future_ts_cols}")

        self.features = [col for col in self.df.columns if col not in exclude_cols]

        # Remove columns with too many missing values
        self.features = [col for col in self.features
                         if self.df[col].isnull().sum() < len(self.df) * 0.5]

        # Only keep numeric columns
        numeric_features = []
        for col in self.features:
            if self.df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                numeric_features.append(col)

        self.features = numeric_features
        print(f"Selected {len(self.features)} historical features: {self.features}")

        if len(self.features) == 0:
            print("No suitable features found. Creating dummy features.")
            self.df['dummy_feature'] = np.random.randn(len(self.df))
            self.features = ['dummy_feature']

        # Create feature matrix
        X = self.df[self.features].fillna(method='ffill').fillna(0)
        y = self.df['target'].fillna(0)

        # Remove rows where target is NaN
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]

        # Remove rows where target is 0 (no price change data)
        if y.std() > 0:  # Only if there's actual variation
            non_zero_idx = y != 0
            if non_zero_idx.sum() > 1000:  # Keep only if we have enough data
                X = X[non_zero_idx]
                y = y[non_zero_idx]

        print(f"Final dataset shape: X={X.shape}, y={y.shape}")
        print(f"Target statistics: mean={y.mean():.6f}, std={y.std():.6f}, min={y.min():.6f}, max={y.max():.6f}")

        return X, y

    def train_model(self, X, y):
        """Train the prediction model"""
        print("Training model...")

        # Use a faster model for large datasets
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor

        if len(X) > 5000:
            # Use Ridge regression for speed
            self.model = Ridge(alpha=1.0, random_state=42)
            print("Using Ridge regression for faster training")
        else:
            # Use Random Forest for smaller datasets
            self.model = RandomForestRegressor(
                n_estimators=50,  # Reduced from 100
                max_depth=10,  # Limit depth
                random_state=42,
                n_jobs=-1
            )
            print("Using Random Forest")

        # Simpler cross-validation for speed
        tscv = TimeSeriesSplit(n_splits=3)  # Reduced from 5

        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # Train and evaluate
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_val_scaled)
            score = r2_score(y_val, y_pred)
            cv_scores.append(score)

            print(f"Fold completed. R2: {score:.4f}")

        print(f"Cross-validation R2 scores: {cv_scores}")
        print(f"Average CV R2: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

        # Final training on all data
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        return self.model

    def evaluate_model(self, X, y):
        """Evaluate model performance"""
        print("Evaluating model...")

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = self.model.predict(X_test_scaled)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Test Results:")
        print(f"R2 Score: {r2:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))

        return {
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'feature_importance': feature_importance
        }

    def plot_results(self, results):
        """Plot model results"""
        plt.figure(figsize=(15, 5))

        # Feature importance plot
        plt.subplot(1, 2, 1)
        top_features = results['feature_importance'].head(10)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 10 Feature Importances')
        plt.gca().invert_yaxis()

        # Prediction vs actual (sample)
        plt.subplot(1, 2, 2)
        X, y = self.prepare_features()
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)

        plt.scatter(y, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Predictions vs Actual')

        plt.tight_layout()
        plt.show()

    def run_pipeline(self):
        """Run the complete pipeline"""
        print("=" * 50)
        print("STOCK PREDICTION PIPELINE")
        print("=" * 50)

        # Load and explore data
        self.load_data()
        self.explore_data()

        # Preprocess data
        self.preprocess_data()

        # Prepare features
        X, y = self.prepare_features()

        # Train model
        self.train_model(X, y)

        # Evaluate model
        results = self.evaluate_model(X, y)

        # Plot results
        self.plot_results(results)

        return results


# Usage example
if __name__ == "__main__":
    # Fast configuration for testing
    pipeline = StockPredictionPipeline(
        sample_size=5000,  # Use smaller sample for speed
        use_sentiment=False  # Skip sentiment analysis for speed
    )
    results = pipeline.run_pipeline()

    # For full dataset with sentiment analysis:
    # pipeline = StockPredictionPipeline(sample_size=50000, use_sentiment=True)
    # results = pipeline.run_pipeline()