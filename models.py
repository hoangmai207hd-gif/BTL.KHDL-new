import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib


# =========================
# LOAD DATA
# =========================

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from an Excel or CSV file."""
    if not isinstance(path, str):
        raise TypeError('Path must be a string.')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")

    extension = os.path.splitext(path)[1].lower()
    if extension in ('.xls', '.xlsx'):
        return pd.read_excel(path)
    if extension == '.csv':
        return pd.read_csv(path)

    raise ValueError('Định dạng file không được hỗ trợ. Vui lòng dùng .xlsx, .xls hoặc .csv')


# =========================
# PREPROCESS DATA
# =========================

def validate_data(df: pd.DataFrame, target_col: str = 'Financial_Status'):
    """Validate that the DataFrame contains the expected target and enough data."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame.')
    if df.empty:
        raise ValueError('The dataset is empty.')
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is missing from the dataset.")
    if df[target_col].isnull().any():
        raise ValueError(f"Target column '{target_col}' contains missing values.")
    if df.drop(columns=[target_col]).empty:
        raise ValueError('No feature columns were found in the dataset.')


def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features and return a numeric feature matrix."""
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=float)
    return X


def preprocess_data(df: pd.DataFrame, target_col: str = 'Financial_Status'):
    """Clean dataset, validate it, encode the target label, and encode categorical features."""
    df = df.copy()

    # Remove duplicate rows and rows with missing values
    df = df.drop_duplicates()
    df = df.dropna()

    validate_data(df, target_col)

    encoder = LabelEncoder()
    df[target_col] = encoder.fit_transform(df[target_col])

    X = df.drop(target_col, axis=1)
    X = encode_features(X)
    y = df[target_col]

    return X, y, encoder


# =========================
# TRAIN MODELS
# =========================

def train_models(X: pd.DataFrame, y: pd.Series, model_path: str = 'best_model.pkl'):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        trained_models[name] = model

    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = trained_models[best_model_name]

    save_model(best_model, model_path)

    return {
        'results': results,
        'trained_models': trained_models,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'X_test': X_test,
        'y_test': y_test
    }


# =========================
# SAVE MODEL
# =========================

def save_model(model, filename: str = 'best_model.pkl'):
    """Save a trained model to disk."""
    joblib.dump(model, filename)


# =========================
# LOAD MODEL
# =========================

def load_model(filename: str = 'best_model.pkl'):
    """Load a trained model from disk."""
    return joblib.load(filename)


# =========================
# FEATURE IMPORTANCE
# =========================

def get_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return None

    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    return feature_df.sort_values(by='Importance', ascending=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train financial status classification models.')
    parser.add_argument('path', help='Path to the Excel dataset file')
    parser.add_argument('--model-path', default='best_model.pkl', help='Path to save the best model')
    args = parser.parse_args()

    data = load_data(args.path)
    X, y, encoder = preprocess_data(data)
    output = train_models(X, y, model_path=args.model_path)

    print('Best model:', output['best_model_name'])
    for name, metrics in output['results'].items():
        print(f"\n{name} accuracy: {metrics['accuracy']:.4f}")
        print(classification_report(y_true=output['y_test'], y_pred=output['trained_models'][name].predict(output['X_test'])))
