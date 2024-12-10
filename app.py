# Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Step 1: Load Dataset
# Sample dataset for demonstration
data = {
    "ResourceID": [1, 2, 3, 4],
    "Title": ["Algebra Basics", "Advanced Algebra", "Physics Fundamentals", "Python for Beginners"],
    "Category": ["Algebra", "Algebra", "Physics", "Programming"],
    "Format": ["Video", "PDF", "Video", "Interactive"],
    "Link": [
        "https://khanacademy.org/algebra-basics",
        "https://example.com/advanced-algebra.pdf",
        "https://youtube.com/physics-fundamentals",
        "https://interactive-python.com",
    ]
}
df_resources = pd.DataFrame(data)

# Step 2: Build the Recommendation System
def recommend_resources(weak_area, preferred_format):
    # Filter by weak area and preferred format
    filtered_resources = df_resources[
        (df_resources["Category"].str.contains(weak_area, case=False)) &
        (df_resources["Format"].str.contains(preferred_format, case=False))
    ]
    
    # If no direct match, use similarity-based recommendation
    if filtered_resources.empty:
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(df_resources["Category"])
        query_vector = tfidf_vectorizer.transform([weak_area])
        
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_indices = similarity_scores.argsort()[-3:][::-1]
        return df_resources.iloc[top_indices][["Title", "Format", "Link"]]
    else:
        return filtered_resources[["Title", "Format", "Link"]]

# Step 3: Build the Web Interface
def main():
    st.title("Personalized Learning Recommendation System")
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
            recommendations = recommend_resources(weak_area, preferred_format)
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
