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
    page_title="Система анализа отзывов",
    page_icon="📊",
    layout="wide"
)


# -------------------------------------------------------
# Text cleaning function
# -------------------------------------------------------
def clean_text(text):
    """
    Очищает текст отзыва:
    - переводит текст в нижний регистр
    - удаляет специальные символы
    - убирает лишние пробелы
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------------------------------------------
# Topic detection
# -------------------------------------------------------
def detect_topic(review):
    """
    Определяет основную тему отзыва по ключевым словам.
    """
    review = clean_text(review)

    topic_keywords = {
        "Сервис": [
            "сервис", "обслуживание", "персонал", "официант",
            "менеджер", "поддержка", "сотрудник", "вежливый",
            "грубый", "отношение"
        ],
        "Доставка": [
            "доставка", "курьер", "опоздал", "поздно", "быстро",
            "заказ", "привезли", "доставили", "время", "задержка",
            "задержалась"
        ],
        "Цена": [
            "цена", "дорого", "дешево", "стоимость", "деньги",
            "переплата", "скидка", "акция", "ценник"
        ],
        "Еда": [
            "еда", "вкус", "вкусно", "невкусно", "блюдо",
            "меню", "порция", "напиток", "обед", "ужин"
        ],
        "Качество": [
            "качество", "свежий", "свежая", "свежее",
            "плохой", "хороший", "испорченный", "брак",
            "нормальный", "отличный"
        ],
        "Чистота": [
            "чисто", "грязно", "чистота", "упаковка",
            "гигиена", "аккуратно", "мусор", "пятна",
            "грязный"
        ],
    }

    detected_topics = []

    for topic, keywords in topic_keywords.items():
        for keyword in keywords:
            if keyword in review:
                detected_topics.append(topic)
                break

    if len(detected_topics) == 0:
        return "Общее"

    return ", ".join(detected_topics)


# -------------------------------------------------------
# Sentiment prediction with neutral class
# -------------------------------------------------------
def predict_with_neutral(model, vectorizer, text, threshold=0.55):
    """
    Определяет тональность отзыва.
    Если уверенность модели ниже порога, отзыв считается нейтральным.
    """
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])

    probabilities = model.predict_proba(vectorized)[0]
    max_probability = probabilities.max()

    if max_probability < threshold:
        return "нейтральный"

    prediction = model.predict(vectorized)[0]
    return prediction


# -------------------------------------------------------
# Default Russian dataset
# -------------------------------------------------------
@st.cache_data
def load_default_dataset():
    data = {
        "Review": [
            "Еда была вкусной, а доставка быстрой",
            "Очень плохой сервис и холодная еда",
            "Цена нормальная, но доставка немного задержалась",
            "Мне понравилось качество и чистая упаковка",
            "Курьер был грубым, заказ привезли поздно",
            "Обслуживание было обычным, ничего особенного",
            "Отличный вкус и вежливый персонал",
            "Еда слишком дорогая для такого качества",
            "Чистое место и хорошее обслуживание",
            "Доставка была приемлемой, но еда средняя",
            "Я очень доволен заказом",
            "Качество продукта было плохим, я разочарован",
            "Время доставки было нормальным",
            "Отличный сервис и свежая еда",
            "Заказ приехал поздно и был неполным",
            "Персонал был очень внимательным и доброжелательным",
            "Упаковка была грязной и неаккуратной",
            "Цена высокая, но качество хорошее",
            "Меню достаточно обычное, без особых впечатлений",
            "Курьер доставил заказ быстро и аккуратно",
            "Блюдо было холодным и невкусным",
            "Сервис хороший, но цена немного завышена",
            "Все было нормально, ничего плохого сказать не могу",
            "Заказ пришел вовремя, еда свежая",
            "Мне не понравилось обслуживание",
            "Доставка была быстрой и удобной",
            "Еда была испорченной",
            "Цена приемлемая, сервис обычный",
            "Персонал грубый и невнимательный",
            "Упаковка чистая и аккуратная"
        ],
        "Sentiment": [
            "положительный",
            "отрицательный",
            "нейтральный",
            "положительный",
            "отрицательный",
            "нейтральный",
            "положительный",
            "отрицательный",
            "положительный",
            "нейтральный",
            "положительный",
            "отрицательный",
            "нейтральный",
            "положительный",
            "отрицательный",
            "положительный",
            "отрицательный",
            "нейтральный",
            "нейтральный",
            "положительный",
            "отрицательный",
            "нейтральный",
            "нейтральный",
            "положительный",
            "отрицательный",
            "положительный",
            "отрицательный",
            "нейтральный",
            "отрицательный",
            "положительный"
        ]
    }

    return pd.DataFrame(data)


# -------------------------------------------------------
# Model training
# -------------------------------------------------------
def train_model(df):
    """
    Обучает модель классификации тональности отзывов.
    Используются CountVectorizer и Multinomial Naive Bayes.
    """
    df = df.copy()

    if "Review" not in df.columns or "Sentiment" not in df.columns:
        df = load_default_dataset()

    df = df[["Review", "Sentiment"]]

    # Remove empty rows and NaN values
    df = df.dropna(subset=["Review", "Sentiment"])
    df["Review"] = df["Review"].astype(str)
    df["Sentiment"] = df["Sentiment"].astype(str)

    df = df[df["Review"].str.strip() != ""]
    df = df[df["Sentiment"].str.strip() != ""]

    # Use default dataset if uploaded/local CSV is broken
    if len(df) < 6:
        df = load_default_dataset()

    df["clean_review"] = df["Review"].apply(clean_text)

    X = df["clean_review"]
    y = df["Sentiment"]

    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized,
            y,
            test_size=0.3,
            random_state=42
        )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )
    matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)

    return model, vectorizer, accuracy, report, matrix


# -------------------------------------------------------
# CSV analysis
# -------------------------------------------------------
def analyze_reviews(df, model, vectorizer):
    """
    Анализирует отзывы из CSV-файла.
    Файл должен содержать колонку Review.
    """
    df = df.copy()

    if "Review" not in df.columns:
        st.error("CSV-файл должен содержать колонку с названием 'Review'.")
        return None

    df = df.dropna(subset=["Review"])
    df["Review"] = df["Review"].astype(str)
    df = df[df["Review"].str.strip() != ""]

    if df.empty:
        st.error("CSV-файл не содержит отзывов для анализа.")
        return None

    df["Очищенный текст"] = df["Review"].apply(clean_text)
    df["Предсказанная тональность"] = df["Review"].apply(
        lambda text: predict_with_neutral(model, vectorizer, text)
    )
    df["Определенная тема"] = df["Review"].apply(detect_topic)

    return df


# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
st.sidebar.title("Review Analytics System")
st.sidebar.write("AI-система для анализа клиентских отзывов")

st.sidebar.markdown("### Возможности")
st.sidebar.write("- Анализ тональности")
st.sidebar.write("- Определение темы отзыва")
st.sidebar.write("- Загрузка CSV-файла")
st.sidebar.write("- Графики и метрики")
st.sidebar.write("- Скачивание результатов")

st.sidebar.markdown("### Темы отзывов")
st.sidebar.write("Сервис, доставка, цена, еда, качество, чистота")


# -------------------------------------------------------
# Main title
# -------------------------------------------------------
st.title("Система анализа клиентских отзывов")

st.write(
    "Это веб-приложение анализирует клиентские отзывы с использованием "
    "методов обработки естественного языка и машинного обучения. "
    "Система определяет тональность отзыва: положительную, отрицательную "
    "или нейтральную, а также выявляет основную тему отзыва."
)


# -------------------------------------------------------
# Load training dataset
# -------------------------------------------------------
default_df = load_default_dataset()

try:
    csv_df = pd.read_csv("reviews.csv")

    if not csv_df.empty and "Review" in csv_df.columns and "Sentiment" in csv_df.columns:
        training_df = csv_df
    else:
        training_df = default_df

except Exception:
    training_df = default_df


# -------------------------------------------------------
# Train model
# -------------------------------------------------------
model, vectorizer, accuracy, report, matrix = train_model(training_df)


# -------------------------------------------------------
# Tabs
# -------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Анализ одного отзыва",
    "Анализ CSV-файла",
    "Оценка модели",
    "Обучающий датасет"
])


# -------------------------------------------------------
# Tab 1: Single review analysis
# -------------------------------------------------------
with tab1:
    st.header("Анализ одного отзыва")

    user_review = st.text_area(
        "Введите отзыв клиента:",
        placeholder="Например: Доставка была поздней, но еда оказалась вкусной."
    )

    if st.button("Проанализировать отзыв"):
        if user_review.strip() == "":
            st.warning("Пожалуйста, введите текст отзыва.")
        else:
            sentiment = predict_with_neutral(model, vectorizer, user_review)
            topic = detect_topic(user_review)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Тональность отзыва", sentiment.capitalize())

            with col2:
                st.metric("Определенная тема", topic)

            st.subheader("Очищенный текст")
            st.write(clean_text(user_review))


# -------------------------------------------------------
# Tab 2: CSV analysis
# -------------------------------------------------------
with tab2:
    st.header("Анализ отзывов из CSV-файла")

    uploaded_file = st.file_uploader(
        "Загрузите CSV-файл с колонкой 'Review'",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            analyzed_df = analyze_reviews(uploaded_df, model, vectorizer)

            if analyzed_df is not None:
                st.subheader("Результаты анализа")
                st.dataframe(analyzed_df)

                st.subheader("Распределение тональности отзывов")
                sentiment_counts = (
                    analyzed_df["Предсказанная тональность"]
                    .value_counts()
                    .reset_index()
                )
                sentiment_counts.columns = ["Тональность", "Количество"]

                fig_sentiment = px.pie(
                    sentiment_counts,
                    names="Тональность",
                    values="Количество",
                    title="Доля отзывов по тональности"
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)

                st.subheader("Распределение тем отзывов")
                topic_counts = (
                    analyzed_df["Определенная тема"]
                    .value_counts()
                    .reset_index()
                )
                topic_counts.columns = ["Тема", "Количество"]

                fig_topic = px.bar(
                    topic_counts,
                    x="Тема",
                    y="Количество",
                    title="Основные темы клиентских отзывов"
                )
                st.plotly_chart(fig_topic, use_container_width=True)

                csv_result = analyzed_df.to_csv(index=False).encode("utf-8-sig")

                st.download_button(
                    label="Скачать результаты анализа в CSV",
                    data=csv_result,
                    file_name="results_review_analysis.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error("Не удалось прочитать CSV-файл. Проверьте структуру файла.")
            st.write("Ошибка:", e)
    else:
        st.info("Загрузите CSV-файл, чтобы начать пакетный анализ отзывов.")


# -------------------------------------------------------
# Tab 3: Model evaluation
# -------------------------------------------------------
with tab3:
    st.header("Оценка модели машинного обучения")

    st.metric("Точность модели", f"{accuracy:.2f}")

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
    st.header("Обучающий датасет")

    dataset_view = training_df.copy()

    if "Review" not in dataset_view.columns or "Sentiment" not in dataset_view.columns:
        dataset_view = default_df

    dataset_view = dataset_view.dropna(subset=["Review", "Sentiment"])
    dataset_view["Очищенный текст"] = dataset_view["Review"].apply(clean_text)
    dataset_view["Определенная тема"] = dataset_view["Review"].apply(detect_topic)

    st.dataframe(dataset_view)

    st.subheader("Краткая информация о датасете")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Всего отзывов", len(dataset_view))

    with col2:
        st.metric("Классы тональности", dataset_view["Sentiment"].nunique())

    with col3:
        st.metric("Количество тем", dataset_view["Определенная тема"].nunique())