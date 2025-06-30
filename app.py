import streamlit as st
import pandas as pd
import numpy as np
import re
import openai

from firebase_auth import init_firebase, verify_user
from analysis_utils import *
from utils_text import *
from analysis_utils import t_test_analysis

# ✅ Init Firebase
init_firebase()

# ✅ Streamlit Page Setup
st.set_page_config(page_title="📊 Smart Data Analyzer", layout="wide")
st.title("🧠 Client Feedback Analyzer")

# ✅ Login Panel
st.sidebar.header("🔐 Login")
email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")
login = st.sidebar.button("Sign In")

login_successful = False
if login:
    success, msg = verify_user(email, password)
    if success:
        st.success(msg)
        login_successful = True
    else:
        st.error(msg)

if not login_successful:
    st.info("Please log in to access the analyzer.")
    st.stop()  # 👈 Giriş yoksa kalan kod çalışmaz

# ✅ OpenAI API Client
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ✅ AI Yorumlama Fonksiyonu
def ai_interpretation(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that analyzes data and provides insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        raw_message = response.choices[0].message.content.strip()
        sentences = re.findall(r'[^.!?]*[.!?]', raw_message)
        return ''.join(sentences).strip()
    except Exception as e:
        return f"**Error during AI interpretation:** {e}"

# ✅ Dosya Yükleme
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV, Excel, JSON, XML, Feather)",
    type=["csv", "xlsx", "xls", "json", "xml", "feather"]
)

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        elif uploaded_file.name.endswith('.xml'):
            df = pd.read_xml(uploaded_file)
        elif uploaded_file.name.endswith('.feather'):
            df = pd.read_feather(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    # ✅ Veri Önizleme
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # ✅ Analiz Seçimi
    option = st.selectbox("Select Analysis Type", [
        "Numeric Summary", "Correlation Matrix", "Chi-Square Test", "T-Test"
    ])

    # 👇 Analizler
    if option == "Numeric Summary":
        result = analyze_numeric(df)
        st.write("### 📊 Descriptive Statistics")
        st.dataframe(result)

        ai_result = ai_interpretation(f"Analyze this:\n{result.to_string()}")
        st.markdown("### 🧠 AI Insights")
        st.write(ai_result)

    elif option == "Correlation Matrix":
        st.write("### 📈 Correlation Matrix")
        fig, corr_df = correlation_plot(df)
        st.plotly_chart(fig, use_container_width=True)

        ai_result = ai_interpretation(f"Explain the matrix:\n{corr_df.to_string()}")
        st.markdown("### 🧠 AI Insights")
        st.write(ai_result)

    elif option == "Chi-Square Test":
        cats = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(cats) >= 2:
            col1 = st.selectbox("Select first categorical column", cats)
            col2 = st.selectbox("Select second categorical column", [c for c in cats if c != col1])

            result, p_val = chi_square_analysis(df, col1, col2)
            st.write(f"**Chi-Square Test Result:** χ² = {result['chi2_stat']:.2f}, p = {result['p_value']:.4f}")
            st.dataframe(result["contingency_table"])

            ai_result = ai_interpretation(f"Chi-square between {col1} and {col2}:\nP={p_val}")
            st.markdown("### 🧠 Data Insights")
            st.write(ai_result)
        else:
            st.warning("Not enough categorical columns.")

    elif option == "T-Test":
        nums = df.select_dtypes(include=np.number).columns.tolist()
        if len(nums) >= 2:
            col1 = st.selectbox("Select first numeric column", nums)
            col2 = st.selectbox("Select second numeric column", [c for c in nums if c != col1])

            try:
                result, p_val = t_test_analysis(df, col1, col2)
                st.write(result)

                ai_result = ai_interpretation(f"T-test result:\nP={p_val}, {col1} vs {col2}")
                st.markdown("### 🧠 AI Insights")
                st.write(ai_result)
            except Exception as e:
                st.error(f"T-Test Error: {e}")
        else:
            st.warning("Not enough numeric columns.")

    st.success("✅ Analysis completed successfully!")
