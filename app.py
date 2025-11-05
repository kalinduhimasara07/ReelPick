import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, render_template, request, jsonify
import requests

# --- Configuration ---
API_KEY = '7b995d3c6fd91a2284b4ad8cb390c7b8' 
BASE_POSTER_URL = "https://image.tmdb.org/t/p/w500"

# --- Data Loading and Recommendation Logic ---

# Load the dataset
try:
    # Assumes the user has placed the file in the 'data' folder
    df = pd.read_csv('data/tmdb_5000_movies.csv')
except FileNotFoundError:
    # Fallback in case the user places it in the root
    df = pd.read_csv('tmdb_5000_movies.csv')

# Preprocessing
df['overview'] = df['overview'].fillna('')

# Initialize the TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Compute the Cosine Similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a Series mapping movie titles to their index
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def fetch_poster_url(movie_id):
    """Fetches poster path from TMDB API using movie ID."""
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}'
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"{BASE_POSTER_URL}{poster_path}"
        return None
    except requests.exceptions.RequestException as e:
        print(f"API Error for movie ID {movie_id}: {e}")
        return None

def get_searched_movie_details(title, df=df, indices=indices):
    """
    Function to get the title, ID, and poster URL for the searched movie.
    """
    if title not in indices.index:
        return None
    
    idx = indices[title]
    movie_id = df['id'].iloc[idx]
    poster_url = fetch_poster_url(movie_id)
    
    return {
        'title': title,
        'poster_url': poster_url if poster_url else 'https://via.placeholder.com/200x300?text=Poster+Not+Found'
    }

def get_recommendations(title, cosine_sim=cosine_sim, df=df, indices=indices):
    """
    Function to get 10 most similar movies, including the poster URL.
    """
    if title not in indices.index:
        return []

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11] # Top 10 similar movies (excluding itself)

    movie_indices = [i[0] for i in sim_scores]

    recommendations = []
    for i, score_tuple in zip(movie_indices, sim_scores):
        movie_id = df['id'].iloc[i] # Get the TMDB ID from the DataFrame
        poster_url = fetch_poster_url(movie_id)
        
        recommendations.append({
            'title': df['title'].iloc[i],
            'similarity': f"{score_tuple[1]*100:.2f}%",
            'poster_url': poster_url if poster_url else 'https://via.placeholder.com/200x300?text=Poster+Not+Found' 
        })

    return recommendations

# --- Flask App Setup ---

app = Flask(__name__)

@app.route('/')
def home():
    """Renders the main search page."""
    return render_template('index.html')

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    """Provides suggestions for the movie search dropdown."""
    search_term = request.args.get('term', '')
    suggestions = df['title'][df['title'].str.contains(search_term, case=False, na=False)].tolist()
    return jsonify(suggestions[:20])

@app.route('/recommend', methods=['GET'])
def recommend():
    """Handles the recommendation request and displays the results."""
    movie_title = request.args.get('movie_title')
    
    if not movie_title:
        return render_template('index.html', error="Please enter a movie title.")

    # 1. Get details for the searched movie
    searched_movie = get_searched_movie_details(movie_title)

    if not searched_movie:
        return render_template('index.html', error=f"Movie '{movie_title}' not found in the dataset.")
    
    # 2. Get recommendations
    recommendations = get_recommendations(movie_title)
    
    if not recommendations:
        # If the movie is found but no recommendations are generated (e.g., overview is empty), 
        # we still show the searched movie card.
        return render_template('index.html', 
                               movie_title=movie_title, 
                               searched_movie=searched_movie,
                               error=f"Movie '{movie_title}' found, but no similar movies could be determined.")
        
    return render_template('index.html', 
                           movie_title=movie_title, 
                           searched_movie=searched_movie,
                           recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)