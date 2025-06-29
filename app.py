import streamlit as st
import pandas as pd
import openai
import docx2txt
import PyPDF2
from io import StringIO
from utils import preprocess_reviews, get_sentiment_summary, generate_swot_analysis

# --- App Config ---
st.set_page_config(page_title="🧠 Client Feedback Analyzer", layout="wide")
st.title("🧠 Client Feedback Analyzer")
st.markdown("Upload your customer feedback files or paste reviews directly. This AI-powered tool will analyze the sentiment, extract key insights, and generate a SWOT analysis.")

# --- Load OpenAI API ---
try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.warning("⚠️ Missing or invalid OpenAI API Key.")
    st.stop()

# --- Input Options ---
input_method = st.radio("Choose input method:", ["📁 Upload File", "📝 Paste Text"])
reviews = ""

if input_method == "📁 Upload File":
    uploaded_file = st.file_uploader("Upload feedback file (PDF, DOCX, CSV, TXT)", type=["pdf", "docx", "csv", "txt"])
    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            reviews = "\n".join([page.extract_text() for page in pdf_reader.pages])
        elif uploaded_file.name.endswith(".docx"):
            reviews = docx2txt.process(uploaded_file)
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            col = st.selectbox("Select column containing reviews:", df.columns)
            reviews = "\n".join(df[col].dropna().astype(str).tolist())
        elif uploaded_file.name.endswith(".txt"):
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            reviews = stringio.read()

elif input_method == "📝 Paste Text":
    reviews = st.text_area("Paste your customer reviews here:")

# --- Analyze Button ---
if st.button("🔍 Analyze Feedback") and reviews.strip():
    with st.spinner("Analyzing reviews..."):

        cleaned_reviews = preprocess_reviews(reviews)
        sentiment_summary = get_sentiment_summary(cleaned_reviews, client)
        swot_result = generate_swot_analysis(cleaned_reviews, client)

        st.subheader("🧭 Sentiment Overview")
        st.write(sentiment_summary)

        st.subheader("📊 SWOT Analysis")
        for key, value in swot_result.items():
            st.markdown(f"**{key}:**")
            st.write(value)

        st.success("✅ Analysis complete. Explore the insights above!")
else:
    st.info("Please upload a file or paste text and click Analyze.")

st.markdown("---")
st.caption("Powered by GPT-4 | Developed by Mel Ozyurt")
