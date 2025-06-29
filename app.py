import streamlit as st
import pandas as pd
import openai
import docx2txt
import PyPDF2
import re
from io import StringIO

from utils.feedback_processor import preprocess_reviews
from utils.swot_engine import generate_swot_analysis
from utils.sentiment_analysis import get_sentiment_summary

# --- App Config ---
st.set_page_config(page_title="🧠 Client Feedback Analyzer", layout="wide")
st.title("🧠 Client Feedback Analyzer")
st.markdown("""
Upload your customer feedback files or paste reviews directly.

This AI-powered tool will:
- Analyze sentiment
- Detect patterns & anomalies
- Generate SWOT analysis for deep insights
""")

# --- Load OpenAI API ---
client = None
try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error("❌ OpenAI API Key missing or invalid. Please check your Streamlit secrets.")

# --- Input Method Selection ---
input_method = st.radio("Choose input method:", ["📁 Upload File", "📝 Paste Text"])
reviews = ""

# --- Handle File Upload ---
if input_method == "📁 Upload File":
    uploaded_file = st.file_uploader("Upload feedback file (PDF, DOCX, CSV, TXT)", type=["pdf", "docx", "csv", "txt"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".pdf"):
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                reviews = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

            elif uploaded_file.name.endswith(".docx"):
                reviews = docx2txt.process(uploaded_file)

            elif uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                col = st.selectbox("Select the column containing customer reviews:", df.columns)
                reviews = "\n".join(df[col].dropna().astype(str).tolist())

            elif uploaded_file.name.endswith(".txt"):
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                reviews = stringio.read()

        except Exception as e:
            st.error(f"❌ Failed to read file: {e}")

# --- Handle Paste Input ---
elif input_method == "📝 Paste Text":
    reviews = st.text_area("Paste your customer reviews here:")

# --- Analyze Button ---
if st.button("🔍 Analyze Feedback") and reviews.strip():
    if client is None:
        st.warning("Please configure your OpenAI API key in Streamlit secrets to run analysis.")
    else:
        with st.spinner("Analyzing reviews..."):

            # --- Preprocessing ---
            cleaned_reviews = preprocess_reviews(reviews)

            # --- Sentiment Analysis ---
            sentiment_summary = get_sentiment_summary(cleaned_reviews, client)

            # --- SWOT Analysis ---
            swot_result = generate_swot_analysis(cleaned_reviews, client)

            # --- Output ---
            st.subheader("🧭 Sentiment Overview")
            st.write(sentiment_summary)

            st.subheader("📊 SWOT Analysis")
            for key, value in swot_result.items():
                st.markdown(f"**{key}:**")
                st.write(value)

            st.success("✅ Analysis complete. Scroll above to explore insights.")
else:
    st.info("Please upload a file or paste text, then click **Analyze Feedback**.")

# --- Footer ---
st.markdown("---")
st.caption("⚙️ Powered by GPT-4 | Built with ❤️ by [YourName]")
