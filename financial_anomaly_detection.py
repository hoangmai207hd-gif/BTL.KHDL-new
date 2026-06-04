import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

DEFAULT_FILE_CANDIDATES = [
    'data.xls',
    'data.xlsx',
    'ResultTestDataOilSurveyVN.xlsx',
    'data.csv',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phát hiện bất thường trong báo cáo tài chính từ file Excel"
    )
    parser.add_argument(
        "--file",
        "-f",
        help="Đường dẫn tới file dữ liệu Excel",
        required=False,
    )
    return parser.parse_args()


def find_default_file():
    for candidate in DEFAULT_FILE_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


def load_dataset(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {file_path}")
    df = pd.read_excel(file_path)
    if df.empty:
        raise ValueError("File dữ liệu rỗng.")
    return df


def preprocess_data(df: pd.DataFrame):
    if 'Financial_Status' not in df.columns:
        raise KeyError(
            f"Cột 'Financial_Status' không tồn tại trong dữ liệu. "
            f"Các cột hiện có: {', '.join(df.columns)}"
        )

    if df['Financial_Status'].dtype == object:
        df['Financial_Status'] = df['Financial_Status'].map({'Normal': 0, 'Anomaly': 1})

    if df['Financial_Status'].isnull().any():
        raise ValueError("Cột 'Financial_Status' chứa giá trị thiếu.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Financial_Status'])

    X = df.drop('Financial_Status', axis=1)
    if X.empty:
        raise ValueError("Không có biến dự đoán nào ngoài 'Financial_Status'.")

    object_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if object_columns:
        X = pd.get_dummies(X, columns=object_columns, drop_first=True)

    if X.isnull().any().any():
        X = X.fillna(X.median())

    return X, y, label_encoder


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("===== BÁO CÁO KẾT QUẢ =====")
    print(f"Độ chính xác tổng thể: {accuracy * 100:.2f}%")
    print("\nChi tiết các chỉ số:")
    target_names = [str(cls) for cls in label_encoder.classes_]
    print(classification_report(y_test, y_pred, target_names=target_names))


def main():
    args = parse_args()
    if args.file:
        file_path = args.file
    else:
        default_file = find_default_file()
        if default_file:
            file_path = default_file
            print(f"Không truyền --file, dùng tệp mặc định: {file_path}")
        else:
            file_path = input("Nhập đường dẫn file dữ liệu Excel: ").strip()

    df = load_dataset(file_path)
    print(f"\nĐã tải dữ liệu: {len(df)} dòng, {len(df.columns)} cột.\n")

    X, y, label_encoder = preprocess_data(df)
    model, X_test, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test, label_encoder)


if __name__ == '__main__':
    main()
