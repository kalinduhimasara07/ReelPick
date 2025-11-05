import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, render_template, request, jsonify, session 
import requests

# --- Configuration ---
API_KEY = '7b995d3c6fd91a2284b4ad8cb390c7b8' 
BASE_POSTER_URL = "https://image.tmdb.org/t/p/w500"
BASE_BACKDROP_URL = "https://image.tmdb.org/t/p/original"

# --- Data Loading and Recommendation Logic ---

# Load the dataset
try:
    df = pd.read_csv('data/tmdb_5000_movies.csv')
except FileNotFoundError:
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
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"{BASE_POSTER_URL}{poster_path}"
        return None
    except requests.exceptions.RequestException as e:
        print(f"API Error for movie ID {movie_id}: {e}")
        return None

def fetch_movie_details(movie_id):
    """Fetches complete movie details from TMDB API."""
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}'
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Format the data for display
        return {
            'title': data.get('title', 'N/A'),
            'tagline': data.get('tagline', ''),
            'overview': data.get('overview', 'No overview available.'),
            'release_date': data.get('release_date', 'N/A'),
            'runtime': data.get('runtime', 0),
            'vote_average': data.get('vote_average', 0),
            'vote_count': data.get('vote_count', 0),
            'budget': data.get('budget', 0),
            'revenue': data.get('revenue', 0),
            'genres': [g['name'] for g in data.get('genres', [])],
            'poster_path': f"{BASE_POSTER_URL}{data['poster_path']}" if data.get('poster_path') else None,
            'backdrop_path': f"{BASE_BACKDROP_URL}{data['backdrop_path']}" if data.get('backdrop_path') else None,
            'homepage': data.get('homepage', ''),
            'status': data.get('status', 'N/A'),
            'production_companies': [pc['name'] for pc in data.get('production_companies', [])][:3]
        }
    except requests.exceptions.RequestException as e:
        print(f"API Error for movie ID {movie_id}: {e}")
        return None

def get_movie_result_set(title, cosine_sim=cosine_sim, df=df, indices=indices):
    """
    Function to get the searched movie details and its top 10 recommendations.
    Returns a dictionary or None if the movie is not found.
    """
    title = title.strip()
    if not title or title not in indices.index:
        return None

    idx = indices[title]
    searched_movie_id = df['id'].iloc[idx]
    searched_poster_url = fetch_poster_url(searched_movie_id)
    
    searched_movie = {
        'title': title,
        'movie_id': int(searched_movie_id),
        'poster_url': searched_poster_url if searched_poster_url else 'https://via.placeholder.com/200x300?text=Poster+Not+Found'
    }

    # Get recommendations
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11] 

    movie_indices = [i[0] for i in sim_scores]

    recommendations = []
    for i, score_tuple in zip(movie_indices, sim_scores):
        movie_id = df['id'].iloc[i]
        poster_url = fetch_poster_url(movie_id)
        
        recommendations.append({
            'title': df['title'].iloc[i],
            'movie_id': int(movie_id),
            'similarity': f"{score_tuple[1]*100:.2f}%",
            'poster_url': poster_url if poster_url else 'https://via.placeholder.com/200x300?text=Poster+Not+Found' 
        })
        
    return {
        'searched_movie': searched_movie,
        'recommendations': recommendations
    }

# --- Flask App Setup ---

app = Flask(__name__)
app.secret_key = 'your_super_secret_key_here' 

@app.route('/')
def home():
    """Renders the main search page, showing cached data (history) if available."""
    if 'search_history' not in session:
        session['search_history'] = []

    if session['search_history']:
        return render_template('index.html', 
                               app_name="ReelPick", 
                               search_history=session['search_history'],
                               state='browse')
    
    return render_template('index.html', app_name="ReelPick", state='initial')

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    """Provides suggestions for the movie search dropdown."""
    search_term = request.args.get('term', '')
    suggestions = df['title'][df['title'].str.contains(search_term, case=False, na=False)].tolist()
    return jsonify(suggestions[:20])

@app.route('/movie/<int:movie_id>', methods=['GET'])
def get_movie_details(movie_id):
    """API endpoint to fetch detailed movie information."""
    details = fetch_movie_details(movie_id)
    if details:
        return jsonify(details)
    return jsonify({'error': 'Movie not found'}), 404

@app.route('/recommend', methods=['GET'])
def recommend():
    """Handles the recommendation request, calculates results, and APPENDS to session history."""
    movie_title_input = request.args.get('movie_title')
    
    if not movie_title_input:
        return home()

    result_set = get_movie_result_set(movie_title_input)
    
    if 'search_history' not in session:
        session['search_history'] = []
    
    error_message = None
    state = 'initial'

    if result_set:
        session['search_history'].append(result_set)
        session.modified = True 
        
        state = 'results'
        return render_template('index.html', 
                               app_name="ReelPick",
                               search_history=session['search_history'],
                               state=state)
    else:
        error_message = f"Movie '{movie_title_input}' not found in the dataset."
        return render_template('index.html', 
                               app_name="ReelPick",
                               search_history=session['search_history'],
                               error=error_message,
                               state=state)

if __name__ == '__main__':
    app.run(debug=True)