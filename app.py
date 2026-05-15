import re
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -------------------------------------------------------
# Page settings
# -------------------------------------------------------
st.set_page_config(
    page_title="Review Analytics System",
    page_icon="📊",
    layout="wide"
)


# -------------------------------------------------------
# Text preprocessing function
# -------------------------------------------------------
def clean_text(text):
    """
    Cleans customer review text:
    - converts text to lowercase
    - removes special characters
    - removes extra spaces
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -------------------------------------------------------
# Topic detection function
# -------------------------------------------------------
def detect_topic(review):
    """
    Detects the main topic of a review using keywords.
    Categories:
    service, delivery, price, food, quality, cleanliness, general
    """
    review = clean_text(review)

    topic_keywords = {
        "Service": [
            "service", "staff", "waiter", "manager", "support",
            "обслуживание", "персонал", "сервис", "официант"
        ],
        "Delivery": [
            "delivery", "courier", "late", "arrived", "order",
            "доставка", "курьер", "опоздал", "заказ"
        ],
        "Price": [
            "price", "expensive", "cheap", "cost", "money",
            "цена", "дорого", "дешево", "стоимость"
        ],
        "Food": [
            "food", "taste", "meal", "dish", "menu",
            "еда", "вкус", "блюдо", "меню"
        ],
        "Quality": [
            "quality", "fresh", "poor", "good", "bad",
            "качество", "свежий", "плохой", "хороший"
        ],
        "Cleanliness": [
            "clean", "dirty", "hygiene", "packaging",
            "чисто", "грязно", "чистота", "упаковка"
        ],
    }

    detected_topics = []

    for topic, keywords in topic_keywords.items():
        for keyword in keywords:
            if keyword in review:
                detected_topics.append(topic)
                break

    if len(detected_topics) == 0:
        return "General"

    return ", ".join(detected_topics)


# -------------------------------------------------------
# Neutral sentiment extension
# -------------------------------------------------------
def predict_with_neutral(model, vectorizer, text, threshold=0.55):
    """
    Predicts sentiment with neutral class.
    If the model confidence is lower than threshold,
    the review is classified as neutral.
    """
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])

    probabilities = model.predict_proba(vectorized)[0]
    max_probability = probabilities.max()

    if max_probability < threshold:
        return "neutral"

    prediction = model.predict(vectorized)[0]
    return prediction


# -------------------------------------------------------
# Load default dataset
# -------------------------------------------------------
@st.cache_data
def load_default_dataset():
    data = {
        "Review": [
            "The food was delicious and delivery was fast",
            "Very bad service and cold food",
            "The price is normal but delivery was late",
            "I liked the quality and clean packaging",
            "The courier was rude and the order was delayed",
            "The service was okay but nothing special",
            "Great taste and friendly staff",
            "The food was too expensive for this quality",
            "Clean place and good service",
            "Delivery was acceptable but the food was average",
            "I am very satisfied with the order",
            "The product quality was poor and disappointing",
            "The delivery time was normal",
            "Excellent service and fresh food",
            "The order arrived late and incomplete"
        ],
        "Sentiment": [
            "positive",
            "negative",
            "neutral",
            "positive",
            "negative",
            "neutral",
            "positive",
            "negative",
            "positive",
            "neutral",
            "positive",
            "negative",
            "neutral",
            "positive",
            "negative"
        ]
    }

    return pd.DataFrame(data)


# -------------------------------------------------------
# Train model
# -------------------------------------------------------
def train_model(df):
    """
    Trains a sentiment classification model.
    The model uses CountVectorizer and Multinomial Naive Bayes.
    """
    df = df.copy()

    df["clean_review"] = df["Review"].apply(clean_text)

    X = df["clean_review"]
    y = df["Sentiment"]

    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # For small datasets, test_size is kept small
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)

    return model, vectorizer, accuracy, report, matrix


# -------------------------------------------------------
# Analyze uploaded dataset
# -------------------------------------------------------
def analyze_reviews(df, model, vectorizer):
    """
    Applies sentiment prediction and topic detection to uploaded reviews.
    """
    df = df.copy()

    if "Review" not in df.columns:
        st.error("CSV file must contain a column named 'Review'.")
        return None

    df["clean_review"] = df["Review"].apply(clean_text)
    df["Predicted Sentiment"] = df["Review"].apply(
        lambda text: predict_with_neutral(model, vectorizer, text)
    )
    df["Detected Topic"] = df["Review"].apply(detect_topic)

    return df


# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
st.sidebar.title("ReviewMind")
st.sidebar.write("AI-based customer review analysis")

st.sidebar.markdown("### Features")
st.sidebar.write("- Sentiment analysis")
st.sidebar.write("- Topic detection")
st.sidebar.write("- CSV upload")
st.sidebar.write("- Charts and metrics")
st.sidebar.write("- Download results")

st.sidebar.markdown("### Topics")
st.sidebar.write("Service, Delivery, Price, Food, Quality, Cleanliness")


# -------------------------------------------------------
# Main title
# -------------------------------------------------------
st.title("Review Analytics System")
st.write(
    "This application analyzes customer reviews using basic NLP techniques "
    "and machine learning. It classifies reviews as positive, negative, or neutral "
    "and detects the main topic of each review."
)


# -------------------------------------------------------
# Dataset loading
# -------------------------------------------------------
default_df = load_default_dataset()

try:
    csv_df = pd.read_csv("reviews.csv")
    if "Review" in csv_df.columns and "Sentiment" in csv_df.columns:
        training_df = csv_df
    else:
        training_df = default_df
except FileNotFoundError:
    training_df = default_df


# -------------------------------------------------------
# Model training
# -------------------------------------------------------
model, vectorizer, accuracy, report, matrix = train_model(training_df)


# -------------------------------------------------------
# Tabs
# -------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Single Review Analysis",
    "CSV Dataset Analysis",
    "Model Evaluation",
    "Training Dataset"
])


# -------------------------------------------------------
# Tab 1: Single review analysis
# -------------------------------------------------------
with tab1:
    st.header("Analyze a Single Review")

    user_review = st.text_area(
        "Enter customer review:",
        placeholder="Example: The delivery was late but the food was good."
    )

    if st.button("Analyze Review"):
        if user_review.strip() == "":
            st.warning("Please enter a review.")
        else:
            sentiment = predict_with_neutral(model, vectorizer, user_review)
            topic = detect_topic(user_review)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Predicted Sentiment", sentiment.capitalize())

            with col2:
                st.metric("Detected Topic", topic)

            st.subheader("Cleaned Text")
            st.write(clean_text(user_review))


# -------------------------------------------------------
# Tab 2: CSV dataset analysis
# -------------------------------------------------------
with tab2:
    st.header("Analyze Reviews from CSV File")

    uploaded_file = st.file_uploader(
        "Upload CSV file with a column named 'Review'",
        type=["csv"]
    )

    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        analyzed_df = analyze_reviews(uploaded_df, model, vectorizer)

        if analyzed_df is not None:
            st.subheader("Analysis Results")
            st.dataframe(analyzed_df)

            st.subheader("Sentiment Distribution")
            sentiment_counts = analyzed_df["Predicted Sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]

            fig_sentiment = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Count",
                title="Sentiment Share"
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)

            st.subheader("Topic Distribution")
            topic_counts = analyzed_df["Detected Topic"].value_counts().reset_index()
            topic_counts.columns = ["Topic", "Count"]

            fig_topic = px.bar(
                topic_counts,
                x="Topic",
                y="Count",
                title="Detected Review Topics"
            )
            st.plotly_chart(fig_topic, use_container_width=True)

            csv_result = analyzed_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Analysis Results as CSV",
                data=csv_result,
                file_name="review_analysis_results.csv",
                mime="text/csv"
            )
    else:
        st.info("Upload a CSV file to start batch review analysis.")


# -------------------------------------------------------
# Tab 3: Model evaluation
# -------------------------------------------------------
with tab3:
    st.header("Model Evaluation")

    st.metric("Model Accuracy", f"{accuracy:.2f}")

    st.subheader("Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.subheader("Confusion Matrix")
    matrix_df = pd.DataFrame(
        matrix,
        index=model.classes_,
        columns=model.classes_
    )
    st.dataframe(matrix_df)

    fig_matrix = px.imshow(
        matrix_df,
        text_auto=True,
        title="Confusion Matrix"
    )
    st.plotly_chart(fig_matrix, use_container_width=True)


# -------------------------------------------------------
# Tab 4: Training dataset
# -------------------------------------------------------
with tab4:
    st.header("Training Dataset")

    dataset_view = training_df.copy()
    dataset_view["clean_review"] = dataset_view["Review"].apply(clean_text)
    dataset_view["Detected Topic"] = dataset_view["Review"].apply(detect_topic)

    st.dataframe(dataset_view)

    st.subheader("Dataset Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Reviews", len(dataset_view))

    with col2:
        st.metric("Sentiment Classes", dataset_view["Sentiment"].nunique())

    with col3:
        st.metric("Detected Topics", dataset_view["Detected Topic"].nunique())