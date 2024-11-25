# Hybrid Spotify Recommender System

Get personalized music recommendations using a hybrid approach that combines content-based filtering, collaborative filtering, and popularity scoring.

## Features

- **Content-based Filtering**: Recommends songs based on audio features like danceability, energy, tempo, and more.
- **Collaborative Filtering**: Suggests songs based on user-item similarity.
- **Hybrid Recommendations**: Combines content-based, collaborative, and popularity-based recommendations.
- **Playlist Analysis**: Extracts song details and audio features from a Spotify playlist.
- **Interactive UI**: Built using Streamlit for a seamless user experience.
- **Customizable Weights**: Adjust weights for different recommendation strategies.

## Requirements

- Python 3.8+
- Spotify Developer Account
- Spotify API Client ID and Client Secret

## Installation

1. Clone the repository:
   
   git clone https://github.com/your-repo/hybrid-spotify-recommender.git
   cd hybrid-spotify-recommender

2. Run the Streamlit app:

   streamlit run app.py
   Open the app in your browser at http://localhost:8501   


## ⚙️ How to Use
1. Enter your Spotify Client ID and Client Secret in the sidebar.
2. Provide a Spotify Playlist ID or URL to load the playlist.
3. Select a song from the playlist and choose the number of recommendations.
4. Pick a recommendation approach:
    a. Hybrid: Customize weighting factors for content-based, collaborative, and popularity scores.
    b. Content-based: Focuses on similar audio features.
    c. Collaborative: Based on similarity in listening patterns.
5. View the recommended songs along with track details, previews, and Spotify links.   


## 📊 Recommendation Approaches
1. Content-based Filtering:
    - Uses audio features such as danceability, tempo, and energy to find similar tracks.
2. Collaborative Filtering:
    - Computes similarity scores using a user-item matrix.
3. Popularity-based Scoring:
    - Incorporates the track's release date and popularity to adjust recommendations.
4. Hybrid Approach:
    - Combines all three methods with customizable weights.


## 📝 Notes
    - Spotify API Rate Limits: Be mindful of the API limits while testing or running the app.
    - Audio Previews: Some tracks may not have audio previews available.
