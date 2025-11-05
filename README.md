# 🎬 ReelPick: Content-Based Movie Recommender

**Live Site:** [https://reelpick.onrender.com/](https://reelpick.onrender.com/)

ReelPick is a web application that provides instant, personalized movie recommendations based on a movie's plot synopsis. It utilizes **Term Frequency-Inverse Document Frequency (TF-IDF)** and **Cosine Similarity** to compare the textual content of movies and suggest the top 10 most similar films.

---

## ✨ Features

* **Content-Based Filtering:** Recommendations are purely based on movie **overviews** (plot summaries), making them highly relevant to a user's content preference.
* **Search History/Browse:** The homepage acts as a persistent browsing history, displaying cards for all movies previously searched during a session.
* **Real-time Suggestions:** An interactive typeahead (autocomplete) feature assists users in finding movie titles present in the dataset.
* **Visual Results:** Displays movie posters, titles, and a calculated similarity percentage using the **TMDB API**.
* **Production Ready:** Hosted as a production service on Render using Gunicorn, bypassing size limitations of typical serverless environments.

---

## 💻 Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Backend Framework** | **Python (Flask)** | Routing, request handling, session management. |
| **Recommendation Engine** | **Scikit-learn** | Generates the TF-IDF matrix and calculates Cosine Similarity. |
| **Data Processing** | **Pandas** | Data loading and preprocessing (`tmdb_5000_movies.csv`). |
| **External Data** | **Requests / TMDB API** | Fetches real-time movie posters and details via `movie_id`. |
| **Frontend** | **HTML5/CSS3** | Structure and styling (Flexbox/Grid for results). |
| **Interactivity** | **jQuery UI (Autocomplete)** | Provides the dynamic search dropdown menu. |
| **Deployment** | **Render** | Production hosting platform for continuous service. |
| **Web Server** | **Gunicorn** | Production-grade WSGI server. |

---

## 🧠 Recommendation Methodology

The core of ReelPick is the Content-Based Filtering mechanism:

1.  **Data Preparation:** The **`overview`** column of the dataset is chosen as the feature, as it contains the richest descriptive text. Null values are filled, and the text is tokenized.
2.  **TF-IDF Vectorization:** The `TfidfVectorizer` from `scikit-learn` converts the text in the `overview` column into a numerical matrix. This vectorization assigns a weight to each word, emphasizing rare and important plot keywords while downplaying common stop words.
3.  **Cosine Similarity:** The `linear_kernel` (which is a fast method to calculate Cosine Similarity) computes a score between every movie vector in the TF-IDF matrix. This score represents the angle between the vectors, quantifying the mathematical similarity between their content.
4.  **Recommendation Generation:** When a user searches for a movie, its index is found, the 10 highest similarity scores (excluding the movie itself) are retrieved, and the corresponding movie titles and details are returned.

---

## 🚀 Local Setup and Installation

Follow these steps to get a local copy of the project running on your machine.

### Prerequisites

You must have Python 3.x installed.

```bash
# Verify Python installation
python --version
