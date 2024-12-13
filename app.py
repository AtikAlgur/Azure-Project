import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

# Step 1: Load Dataset
st.title("Personalized Learning Recommendation System")

file_path = "resources.xlsx"
try:
    df_resources = pd.read_excel(file_path)
    st.success("Dataset loaded successfully!")
except FileNotFoundError:
    st.error(f"File {file_path} not found. Please upload the correct file.")
    df_resources = None

# Step 2: Train the Dataset
if df_resources is not None:
    # Combine Title and Category for better context
    df_resources["Combined"] = df_resources["Title"].fillna("") + " " + df_resources["Category"].fillna("")
    label_encoder = LabelEncoder()
    df_resources["Encoded_Category"] = label_encoder.fit_transform(df_resources["Category"].fillna(""))

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_resources["Combined"].fillna(""))
    model = MultinomialNB()
    model.fit(tfidf_matrix, df_resources["Encoded_Category"])

def classify_category(weak_area):
    if df_resources is None:
        st.error("Dataset is not loaded. Please check the file path.")
        return ""
    query_vector = vectorizer.transform([weak_area])
    predicted_category = model.predict(query_vector)
    return label_encoder.inverse_transform(predicted_category)[0]

# Step 3: Build the Recommendation System
def recommend_resources(predicted_category, weak_area, preferred_format):
    if df_resources is None:
        st.error("Dataset is not loaded. Please check the file path.")
        return pd.DataFrame()

    # Filter by category, weak area, and preferred format
    filtered_resources = df_resources[
        (df_resources["Category"].str.contains(predicted_category, case=False, na=False)) &
        (df_resources["Title"].str.contains(weak_area, case=False, na=False)) &
        (df_resources["Format"].str.contains(preferred_format, case=False, na=False))
    ]
    
    # If no direct match, use similarity-based recommendation
    if filtered_resources.empty:
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(df_resources["Combined"].fillna(""))
        query_vector = tfidf_vectorizer.transform([weak_area])

        similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_indices = similarity_scores.argsort()[-3:][::-1]
        return df_resources.iloc[top_indices][["Title", "Format", "Link"]]
    else:
        return filtered_resources[["Title", "Format", "Link"]]

# Step 4: Build the Web Interface
def main():
    st.write("Get recommendations for learning resources based on your preferences.")

    # Input Form
    name = st.text_input("Enter your name:")
    weak_area = st.text_input("Enter the subject/topic you are weak in (e.g., Algebra, Physics):")
    preferred_format = st.selectbox(
        "Select your preferred learning format:",
        ["Video", "PDF", "Interactive"]
    )

    # Get Recommendations
    if st.button("Get Recommendations"):
        if not weak_area or not preferred_format:
            st.warning("Please fill in all fields!")
        else:
            category = classify_category(weak_area)
            if category:
                recommendations = recommend_resources(category, weak_area, preferred_format)
                st.subheader(f"Recommendations for {name}:")
                if recommendations.empty:
                    st.write("No suitable resources found. Try a different topic or format!")
                else:
                    for index, row in recommendations.iterrows():
                        st.write(f"**{row['Title']}** ({row['Format']})")
                        st.write(f"[Learn More]({row['Link']})")
                        st.write("---")

if __name__ == "__main__":
    main()
