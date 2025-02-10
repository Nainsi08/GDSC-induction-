import pandas as pd
import numpy as np
import zipfile
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Extract the ratings file
with zipfile.ZipFile("C:/Users/nainsi/Desktop/Ai/rating.csv.zip", "r") as zip_ref:
    zip_ref.extractall()

# Load datasets
anime_data = pd.read_csv("anime.csv")
ratings_data = pd.read_csv("rating.csv")

# Drop missing values in anime dataset
anime_data.dropna(inplace=True)

# Convert genres into a numerical format for comparison
vectorizer = TfidfVectorizer(stop_words='english')
genre_matrix = vectorizer.fit_transform(anime_data['genre'].fillna(''))

# Compute similarity between anime based on genres
genre_similarity = cosine_similarity(genre_matrix, genre_matrix)

# Recommend anime with similar genres

def recommend_by_genre(anime_name):
    anime_index = anime_data[anime_data['name'] == anime_name].index
    if len(anime_index) == 0:
        return "Anime not found."
    
    anime_index = anime_index[0]
    similarity_scores = sorted(enumerate(genre_similarity[anime_index]), key=lambda x: x[1], reverse=True)[1:11]
    
    recommended_indexes = [i[0] for i in similarity_scores]
    return anime_data.iloc[recommended_indexes][['name', 'genre', 'rating']]

# Create a user-item rating matrix
rating_matrix = ratings_data.pivot_table(index='user_id', columns='anime_id', values='rating')

# Recommend anime based on user ratings

def recommend_by_ratings_auto(num_recommendations=10):
    active_users = rating_matrix.dropna(thresh=10).index  
    if len(active_users) == 0:
        return "No active users found."

    selected_user = np.random.choice(active_users)  
    user_ratings = rating_matrix.loc[selected_user].dropna()

    similar_users = rating_matrix.corrwith(user_ratings, axis=1, method='pearson')
    similar_users = similar_users.dropna().sort_values(ascending=False)

    top_users = similar_users.index[1:11]  
    recommended_anime = rating_matrix.loc[top_users].mean().sort_values(ascending=False)
    recommended_anime = recommended_anime.drop(user_ratings.index, errors='ignore')  
    top_anime_ids = recommended_anime.head(num_recommendations).index

    return anime_data[anime_data['anime_id'].isin(top_anime_ids)][['name', 'genre', 'rating']]

# Streamlit Web App
st.title("Anime Recommender System")

option = st.radio("Select Recommendation Type:", ("By Genre", "By User Ratings", "Hybrid"))

if option == "By Genre":
    anime_name = st.selectbox("Select an Anime:", sorted(anime_data['name'].unique()))  
    if st.button("Get Recommendations"):
        st.write(recommend_by_genre(anime_name))

elif option == "By User Ratings":
    if st.button("Get Recommendations"):
        st.write(recommend_by_ratings_auto())

elif option == "Hybrid":
    anime_name = st.selectbox("Select an Anime:", sorted(anime_data['name'].unique()))  
    if st.button("Get Recommendations"):
        st.subheader("Based on Genre")
        st.write(recommend_by_genre(anime_name))

        st.subheader("Based on User Ratings")
        st.write(recommend_by_ratings_auto())
