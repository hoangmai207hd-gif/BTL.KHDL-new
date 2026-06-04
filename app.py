import os
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

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
    page_title='AI Financial Risk Detection',
    page_icon='💰',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
<div style='text-align:center'>
<h1>💰 AI Financial Risk Detection System</h1>
<h4>Financial Statement Anomaly & Risk Classification Platform</h4>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background-color:#0e1117;
}

[data-testid="stHeader"]{
    background:rgba(0,0,0,0);
}

h1,h2,h3,h4{
    color:white;
}

.metric-box{
    background:#1f2937;
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 5px 15px rgba(0,0,0,0.3);
}

.big-number{
    font-size:32px;
    font-weight:bold;
    color:#00ff99;
}

</style>
""", unsafe_allow_html=True)


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
        best_model_summary = next((r for r in results_list if r['Model'] == best_model_name), None)

        if best_model_summary is not None:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric('🏆 Best Model', best_model_name)
            with col2:
                st.metric('🎯 Accuracy', f"{best_model_summary['Accuracy']:.2%}")
            with col3:
                st.metric('📈 Recall', f"{best_model_summary['Recall']:.2%}")
            with col4:
                st.metric('🔥 F1 Score', f"{best_model_summary['F1-Score']:.2%}")

        tab1, tab2, tab3 = st.tabs([
            '📊 Dataset',
            '🤖 Models',
            '⭐ Features'
        ])

        with tab1:
            st.subheader('📁 Dataset overview')
            st.dataframe(df.head(20), use_container_width=True)

            st.subheader('📊 Phân bố dữ liệu')
            fraud_counts = df['Financial_Status'].value_counts()
            fig_dataset = px.bar(
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
            st.plotly_chart(fig_dataset, use_container_width=True)

        with tab2:
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
            st.plotly_chart(fig, use_container_width=True)

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

                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(
                    cm_df,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    ax=ax
                )
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)

        with tab3:
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
    st.title('🚨 Financial Risk Prediction')
    st.markdown(
        'Dự đoán mức độ rủi ro tài chính từ dữ liệu báo cáo tài chính.\n\n'
        'Các mức dự đoán: 🟢 Normal, 🟡 Anomaly, 🔴 High Risk'
    )
    st.divider()

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
        st.divider()

        if st.button('🚀 Bắt đầu dự đoán'):
            try:
                predict_X = align_upload_data(predict_df, X_columns)
                predictions = selected_model.predict(predict_X)
                labels = encoder.inverse_transform(predictions)

                result_df = predict_df.copy()
                result_df['Prediction'] = labels

                normal_count = (labels == 'Normal').sum()
                anomaly_count = (labels == 'Anomaly').sum()
                high_count = (labels == 'High Risk').sum()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric('🟢 Normal', normal_count)
                with col2:
                    st.metric('🟡 Anomaly', anomaly_count)
                with col3:
                    st.metric('🔴 High Risk', high_count)

                st.divider()

                pie_df = pd.DataFrame({
                    'Status': ['Normal', 'Anomaly', 'High Risk'],
                    'Count': [normal_count, anomaly_count, high_count]
                })
                fig_pie = px.pie(
                    pie_df,
                    names='Status',
                    values='Count',
                    hole=0.55,
                    title='📊 Phân bố kết quả dự đoán'
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                risk_score = (high_count / len(labels)) * 100
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode='gauge+number',
                        value=risk_score,
                        title={'text': 'Financial Risk Score'},
                        gauge={'axis': {'range': [0, 100]}}
                    )
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.divider()
                st.subheader('📋 Kết quả dự đoán')

                def color_status(value):
                    if value == 'High Risk':
                        return 'background-color:#ff4b4b;color:white'
                    elif value == 'Anomaly':
                        return 'background-color:#ffa500;color:black'
                    return 'background-color:#00cc66;color:white'

                st.dataframe(
                    result_df.style.map(
                        color_status,
                        subset=['Prediction']
                    ),
                    use_container_width=True
                )

                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    '📥 Download Results',
                    csv,
                    file_name='prediction_results.csv',
                    mime='text/csv'
                )

            except Exception as exc:
                st.error(f'Không thể dự đoán với dữ liệu hiện tại: {exc}')
