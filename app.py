import os
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

from models import (
    load_data,
    preprocess_data,
    train_models,
    get_feature_importance,
    load_model,
    encode_features
)


# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title='AI Phát Hiện Gian Lận Tài Chính',
    page_icon='📊',
    layout='wide'
)


@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    return load_data(path)


@st.cache_resource
def build_models(df: pd.DataFrame):
    X, y, encoder = preprocess_data(df)
    output = train_models(X, y)
    output['encoder'] = encoder
    output['feature_columns'] = X.columns.tolist()
    return output


def align_upload_data(upload_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    upload_X = encode_features(upload_df)
    upload_X = upload_X.reindex(columns=feature_columns, fill_value=0)
    return upload_X


def find_dataset_file(filename: str) -> str | None:
    current_dir = Path(__file__).resolve().parent
    candidate = current_dir / filename
    if candidate.exists():
        return str(candidate)

    candidate = Path.cwd() / filename
    if candidate.exists():
        return str(candidate)

    for parent in current_dir.parents:
        candidate = parent / filename
        if candidate.exists():
            return str(candidate)

    return None


def load_uploaded_or_default_dataset(uploaded_file, default_path: str | None) -> pd.DataFrame:
    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                return pd.read_csv(uploaded_file)
            return pd.read_excel(uploaded_file)
        except Exception as exc:
            raise ValueError(f'Không thể đọc file tải lên: {exc}')

    if default_path is not None:
        return load_dataset(default_path)

    raise FileNotFoundError(
        'Không tìm thấy file dữ liệu mặc định. Vui lòng tải lên file Excel/CSV hoặc đặt tên file đúng.'
    )


# =========================
# LOAD DATA
# =========================

dataset_filename = 'Financial Statement Anomaly Dataset.xlsx'
found_path = find_dataset_file(dataset_filename)

dataset_upload_file = st.sidebar.file_uploader(
    '📂 Upload file dữ liệu mô hình (Excel hoặc CSV)',
    type=['xlsx', 'xls', 'csv']
)

try:
    df = load_uploaded_or_default_dataset(dataset_upload_file, found_path)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if dataset_upload_file is not None:
    st.sidebar.success(f'Đang dùng dữ liệu upload: {dataset_upload_file.name}')
elif found_path is not None:
    st.sidebar.success(f'Tìm thấy file dữ liệu mặc định: {found_path}')


model_output = build_models(df)
X_columns = model_output['feature_columns']


# =========================
# SIDEBAR MENU
# =========================

menu = st.sidebar.radio(
    '📌 Chọn Trang',
    [
        'Tổng quan',
        'Huấn luyện',
        'Dự đoán'
    ]
)


# ==================================================
# PAGE 1 - TỔNG QUAN
# ==================================================
if menu == 'Tổng quan':
    st.header('📁 Tổng Quan Dữ Liệu')

    vietnamese_columns = {
        'Total_Assets': 'Tổng tài sản',
        'Total_Liabilities': 'Tổng nợ phải trả',
        'Revenue': 'Doanh thu',
        'Operating_Expenses': 'Chi phí hoạt động',
        'Net_Income': 'Lợi nhuận ròng',
        'Cash_Flow_Operating': 'Dòng tiền hoạt động',
        'Cash_Flow_Investing': 'Dòng tiền đầu tư',
        'Cash_Flow_Financing': 'Dòng tiền tài chính',
        'Current_Ratio': 'Hệ số thanh toán',
        'Debt_to_Equity': 'Nợ/Vốn chủ sở hữu',
        'Gross_Margin': 'Biên lợi nhuận gộp',
        'Return_on_Assets': 'ROA',
        'Return_on_Equity': 'ROE',
        'Financial_Status': 'Trạng thái tài chính'
    }

    display_df = df.rename(columns=vietnamese_columns)
    st.dataframe(display_df.head(20), use_container_width=True)

    st.subheader('📊 Phân bố dữ liệu')
    fraud_counts = df['Financial_Status'].value_counts()

    fig = px.bar(
        x=fraud_counts.index,
        y=fraud_counts.values,
        color=fraud_counts.index,
        text=fraud_counts.values,
        labels={
            'x': 'Trạng thái',
            'y': 'Số lượng'
        },
        title='Phân bố hồ sơ tài chính'
    )
    st.plotly_chart(fig, use_container_width=True)


# ==================================================
# PAGE 2 - TRAIN MODEL
# ==================================================
elif menu == 'Huấn luyện':
    st.header('🤖 Huấn Luyện Mô Hình AI')

    if st.button('🚀 Bắt đầu huấn luyện'):
        with st.spinner('Đang huấn luyện AI...'):
            results = model_output['results']
            results_list = model_output.get('results_list', [])
            trained_models = model_output['trained_models']
            best_model_name = model_output['best_model_name']
            best_model = model_output['best_model']
            encoder = model_output['encoder']

        st.success(f'✅ Mô hình tốt nhất theo F1-Score: {best_model_name}')
        st.info(
            'Hiệu suất các mô hình được đánh giá thông qua Accuracy, Precision, Recall và F1-Score. '
            'Với dữ liệu mất cân bằng trong bài toán phát hiện gian lận tài chính, '
            'F1-Score được chọn làm chỉ số chính vì nó phản ánh đồng thời khả năng phát hiện đúng gian lận và hạn chế cảnh báo sai.'
        )

        score_df = pd.DataFrame(results_list)

        st.subheader('📋 Bảng đánh giá mô hình')
        st.dataframe(score_df, use_container_width=True)

        fig_score = px.bar(
            score_df,
            x='Model',
            y='Accuracy',
            color='Model',
            text='Accuracy',
            title='So sánh độ chính xác mô hình AI'
        )
        st.plotly_chart(fig_score, use_container_width=True)

        metric_df = score_df.melt(
            id_vars='Model',
            var_name='Metric',
            value_name='Score'
        )

        fig = px.bar(
            metric_df,
            x='Model',
            y='Score',
            color='Metric',
            barmode='group',
            title='So sánh các chỉ số đánh giá'
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        for name, value in results.items():
            report = value['report']
            cm = value['confusion_matrix']
            st.subheader(f'📋 {name}')
            st.dataframe(
                pd.DataFrame(report).transpose(),
                use_container_width=True
            )

            cm_df = pd.DataFrame(
                cm,
                index=encoder.classes_,
                columns=encoder.classes_
            )
            st.markdown('**Ma trận nhầm lẫn**')
            st.dataframe(cm_df, use_container_width=True)

        st.subheader('📌 Mức độ quan trọng của biến')
        feature_df = get_feature_importance(best_model, X_columns)
        if feature_df is not None:
            fig_importance = px.bar(
                feature_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance'
            )
            st.plotly_chart(fig_importance, use_container_width=True)


# ==================================================
# PAGE 3 - PREDICTION
# ==================================================
else:
    st.header('🔍 Dự Đoán Gian Lận')

    trained_models = model_output['trained_models']
    best_model_name = model_output['best_model_name']
    encoder = model_output['encoder']

    selected_model_name = st.selectbox(
        '🤖 Chọn mô hình AI',
        list(trained_models.keys()),
        index=list(trained_models.keys()).index(best_model_name)
    )
    selected_model = trained_models[selected_model_name]

    uploaded_file = st.file_uploader(
        '📂 Upload file dự đoán (Excel hoặc CSV)',
        type=['xlsx', 'xls', 'csv']
    )

    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith('.csv'):
            predict_df = pd.read_csv(uploaded_file)
        else:
            predict_df = pd.read_excel(uploaded_file)
        st.subheader('📄 Dữ liệu tải lên')
        st.dataframe(predict_df.head(), use_container_width=True)

        try:
            predict_X = align_upload_data(predict_df, X_columns)
            predictions = selected_model.predict(predict_X)
            labels = encoder.inverse_transform(predictions)

            predict_df['Kết quả AI'] = labels
            st.subheader('📊 Kết quả dự đoán')
            st.dataframe(predict_df, use_container_width=True)

            pie_data = predict_df['Kết quả AI'].value_counts()
            fig_pie = px.pie(
                names=pie_data.index,
                values=pie_data.values,
                title='Tỷ lệ phân loại hồ sơ'
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            high_risk_count = (predict_df['Kết quả AI'] == 'High Risk').sum()
            if high_risk_count > 0:
                st.error(f'⚠️ Phát hiện {high_risk_count} hồ sơ có nguy cơ gian lận cao!')
            else:
                st.success('✅ Không phát hiện dấu hiệu bất thường!')
        except Exception as exc:
            st.error(f'Không thể dự đoán với dữ liệu hiện tại: {exc}')
