import pandas as pd
import numpy as np
import zipfile
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Unzip the ratings file
with zipfile.ZipFile("C:/Users/nainsi/Desktop/Ai/rating.csv.zip", "r") as zip_ref:
    zip_ref.extractall()

# Step 2: Load datasets
anime_data = pd.read_csv("anime.csv")
ratings_data = pd.read_csv("rating.csv")

# Step 3: Handle missing values
anime_data.dropna(inplace=True)

### Content-Based Filtering (Genre Similarity) ###
# Convert genres into numerical form using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
genre_matrix = vectorizer.fit_transform(anime_data['genre'].fillna(''))

# Compute similarity scores using cosine similarity
genre_similarity = cosine_similarity(genre_matrix, genre_matrix)

# Function to recommend anime based on genre similarity
def recommend_by_genre(anime_name):
    anime_index = anime_data[anime_data['name'] == anime_name].index
    if len(anime_index) == 0:
        return "Anime not found."

    anime_index = anime_index[0]
    similarity_scores = list(enumerate(genre_similarity[anime_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:11]  # Top 10 similar anime
    
    recommended_indexes = [i[0] for i in similarity_scores]
    return anime_data.iloc[recommended_indexes][['name', 'genre', 'rating']]

### Collaborative Filtering (User-Based Recommendations without User ID) ###
# Create a user-item rating matrix
rating_matrix = ratings_data.pivot_table(index='user_id', columns='anime_id', values='rating')

# Function to automatically select a random active user and recommend anime
def recommend_by_ratings_auto(num_recommendations=10):
    # Select a user who has rated at least 10 anime
    active_users = rating_matrix.dropna(thresh=10).index  
    if len(active_users) == 0:
        return "No active users found."

    selected_user = np.random.choice(active_users)  # Select a random active user
    user_ratings = rating_matrix.loc[selected_user].dropna()

    # Find similar users by comparing their rating patterns
    similar_users = rating_matrix.corrwith(user_ratings, axis=1, method='pearson')
    similar_users = similar_users.dropna().sort_values(ascending=False)

    # Get top similar users
    top_users = similar_users.index[1:11]  # Exclude the user itself
    recommended_anime = rating_matrix.loc[top_users].mean().sort_values(ascending=False)  # Average ratings of top users

    recommended_anime = recommended_anime.drop(user_ratings.index, errors='ignore')  # Remove already watched anime
    top_anime_ids = recommended_anime.head(num_recommendations).index

    return anime_data[anime_data['anime_id'].isin(top_anime_ids)][['name', 'genre', 'rating']]

### Streamlit Web App ###
st.title("Anime Recommender System")

option = st.radio("Select Recommendation Type:", ("By Genre", "By User Ratings", "Hybrid"))

if option == "By Genre":
    anime_list = sorted(anime_data['name'].unique())  # Get unique anime names sorted
    anime_name = st.selectbox("Select an Anime:", anime_list)  # Dropdown menu
    
    if st.button("Get Recommendations"):
        recommendations = recommend_by_genre(anime_name)
        st.write(recommendations)

elif option == "By User Ratings":
    if st.button("Get Recommendations"):
        recommendations = recommend_by_ratings_auto()
        st.write(recommendations)

elif option == "Hybrid":
    anime_list = sorted(anime_data['name'].unique())  # Get unique anime names sorted
    anime_name = st.selectbox("Select an Anime:", anime_list)  # Dropdown menu
    
    if st.button("Get Recommendations"):
        genre_recommendations = recommend_by_genre(anime_name)
        rating_recommendations = recommend_by_ratings_auto()

        st.subheader("Based on Genre")
        st.write(genre_recommendations)

        st.subheader("Based on User Ratings")
        st.write(rating_recommendations)

