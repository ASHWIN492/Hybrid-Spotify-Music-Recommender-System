import streamlit as st
import requests
import base64
import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

class HybridSpotifyRecommender:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = self._get_access_token()
        self.sp = spotipy.Spotify(auth=self.access_token)
        self.music_df = None
        self.music_features_scaled = None
        self.user_item_matrix = None
        
    def _get_access_token(self):
        """Get Spotify API access token."""
        client_credentials = f"{self.client_id}:{self.client_secret}"
        client_credentials_base64 = base64.b64encode(client_credentials.encode()).decode()
        
        token_url = 'https://accounts.spotify.com/api/token'
        headers = {'Authorization': f'Basic {client_credentials_base64}'}
        data = {'grant_type': 'client_credentials'}
        
        response = requests.post(token_url, data=data, headers=headers)
        if response.status_code == 200:
            return response.json()['access_token']
        else:
            raise Exception("Failed to obtain access token")

    def train_on_playlist(self, playlist_id):
        """Train the recommender on playlist data."""
        try:
            # Fetch tracks from playlist
            results = self.sp.playlist_tracks(playlist_id)
            tracks = results['items']
            while results['next']:
                results = self.sp.next(results)
                tracks.extend(results['items'])

            # Process track data
            music_data = []
            for item in tracks:
                if item['track'] is None:
                    continue
                
                track = item['track']
                audio_features = self.sp.audio_features(track['id'])
                if not audio_features or not audio_features[0]:
                    continue
                
                features = audio_features[0]
                
                # Get track data
                track_data = {
                    'Track ID': track['id'],
                    'Track Name': track['name'],
                    'Artists': ', '.join([artist['name'] for artist in track['artists']]),
                    'Album Name': track['album']['name'],
                    'Release Date': track['album']['release_date'],
                    'Popularity': track['popularity'],
                    'Preview URL': track['preview_url'],
                    'External URL': track['external_urls']['spotify'],
                    'Duration (ms)': features['duration_ms'],
                    'Danceability': features['danceability'],
                    'Energy': features['energy'],
                    'Key': features['key'],
                    'Loudness': features['loudness'],
                    'Mode': features['mode'],
                    'Speechiness': features['speechiness'],
                    'Acousticness': features['acousticness'],
                    'Instrumentalness': features['instrumentalness'],
                    'Liveness': features['liveness'],
                    'Valence': features['valence'],
                    'Tempo': features['tempo']
                }
                music_data.append(track_data)

            self.music_df = pd.DataFrame(music_data)
            
            # Scale features
            feature_cols = ['Danceability', 'Energy', 'Key', 'Loudness', 'Mode',
                          'Speechiness', 'Acousticness', 'Instrumentalness',
                          'Liveness', 'Valence', 'Tempo']
            
            scaler = MinMaxScaler()
            self.music_features_scaled = scaler.fit_transform(self.music_df[feature_cols])
            
            # Create user-item matrix for collaborative filtering
            self._create_user_item_matrix()
            
            return self.music_df
            
        except Exception as e:
            raise Exception(f"Error training on playlist: {str(e)}")

    def _create_user_item_matrix(self):
        """Create user-item matrix based on audio features similarity."""
        similarity_matrix = cosine_similarity(self.music_features_scaled)
        self.user_item_matrix = pd.DataFrame(
            similarity_matrix,
            index=self.music_df['Track Name'],
            columns=self.music_df['Track Name']
        )

    def _calculate_weighted_popularity(self, release_date):
        """Calculate time-weighted popularity score."""
        release_date = datetime.strptime(release_date[:10], '%Y-%m-%d')
        time_span = datetime.now() - release_date
        return 1 / (time_span.days + 1)

    def get_content_based_recommendations(self, input_song_name, num_recommendations=5):
        """Get content-based recommendations."""
        if input_song_name not in self.music_df['Track Name'].values:
            raise Exception(f"'{input_song_name}' not found in the playlist")
            
        input_idx = self.music_df[self.music_df['Track Name'] == input_song_name].index[0]
        similarity_scores = cosine_similarity([self.music_features_scaled[input_idx]], 
                                           self.music_features_scaled)
        
        similar_indices = similarity_scores[0].argsort()[::-1][1:num_recommendations+1]
        recommendations = self.music_df.iloc[similar_indices].copy()
        recommendations['Similarity Score'] = similarity_scores[0][similar_indices]
        
        return recommendations

    def get_collaborative_recommendations(self, input_song_name, num_recommendations=5):
        """Get collaborative filtering recommendations."""
        if input_song_name not in self.user_item_matrix.index:
            raise Exception(f"'{input_song_name}' not found in the playlist")
            
        similar_songs = self.user_item_matrix.loc[input_song_name].sort_values(ascending=False)[1:num_recommendations+1]
        recommendations = self.music_df[self.music_df['Track Name'].isin(similar_songs.index)].copy()
        recommendations['Similarity Score'] = similar_songs.values
        
        return recommendations

    def get_hybrid_recommendations(self, input_song_name, num_recommendations=5, 
                                 content_weight=0.4, collab_weight=0.4, popularity_weight=0.2):
        """Get hybrid recommendations combining all approaches."""
        if input_song_name not in self.music_df['Track Name'].values:
            raise Exception(f"'{input_song_name}' not found in the playlist")
            
        # Get content-based recommendations
        content_recs = self.get_content_based_recommendations(
            input_song_name, 
            num_recommendations=num_recommendations*2
        )
        
        # Get collaborative recommendations
        collab_recs = self.get_collaborative_recommendations(
            input_song_name, 
            num_recommendations=num_recommendations*2
        )
        
        # Combine recommendations
        all_songs = pd.concat([
            content_recs[['Track Name', 'Similarity Score']].rename(
                columns={'Similarity Score': 'Content Score'}
            ),
            collab_recs[['Track Name', 'Similarity Score']].rename(
                columns={'Similarity Score': 'Collaborative Score'}
            )
        ])
        
        # Calculate popularity scores
        all_songs = all_songs.merge(
            self.music_df[['Track Name', 'Popularity', 'Release Date']], 
            on='Track Name'
        )
        
        # Calculate time-weighted popularity
        all_songs['Weighted Popularity'] = all_songs.apply(
            lambda x: x['Popularity'] * self._calculate_weighted_popularity(x['Release Date']),
            axis=1
        )
        
        # Normalize scores
        for column in ['Content Score', 'Collaborative Score', 'Weighted Popularity']:
            if column in all_songs.columns:
                all_songs[column] = (all_songs[column] - all_songs[column].min()) / \
                                  (all_songs[column].max() - all_songs[column].min())
        
        # Calculate hybrid score
        all_songs['Hybrid Score'] = (
            content_weight * all_songs['Content Score'].fillna(0) +
            collab_weight * all_songs['Collaborative Score'].fillna(0) +
            popularity_weight * all_songs['Weighted Popularity']
        )
        
        # Get top recommendations
        final_recommendations = all_songs.sort_values(
            'Hybrid Score', 
            ascending=False
        ).drop_duplicates('Track Name').head(num_recommendations)
        
        # Get full song details
        recommendations = self.music_df[
            self.music_df['Track Name'].isin(final_recommendations['Track Name'])
        ].merge(final_recommendations[['Track Name', 'Hybrid Score']], on='Track Name')
        
        return recommendations.sort_values('Hybrid Score', ascending=False)

def main():
    st.set_page_config(page_title="Hybrid Spotify Recommender", 
                      page_icon="🎵", 
                      layout="wide")
    
    st.title("🎵 Hybrid Spotify Music Recommender")
    st.write("Get personalized music recommendations using a hybrid approach!")

    # Sidebar for credentials
    st.sidebar.header("API Credentials")
    client_id = st.sidebar.text_input("Spotify Client ID", type="password")
    client_secret = st.sidebar.text_input("Spotify Client Secret", type="password")

    if not (client_id and client_secret):
        st.warning("Please enter your Spotify API credentials in the sidebar.")
        return

    try:
        recommender = HybridSpotifyRecommender(client_id, client_secret)
        
        # Playlist input
        st.header("1. Select Playlist")
        playlist_id = st.text_input(
            "Enter Spotify Playlist ID or URL",
            help="Example: spotify:playlist:37i9dQZF1DX76Wlfdnj7AP or full Spotify URL"
        )
        
        if playlist_id:
            # Extract playlist ID from URL if needed
            if "spotify.com" in playlist_id:
                playlist_id = playlist_id.split("/")[-1].split("?")[0]
            elif "spotify:playlist:" in playlist_id:
                playlist_id = playlist_id.split(":")[-1]

            # Train button
            if st.button("Load Playlist"):
                with st.spinner("Loading and analyzing playlist..."):
                    music_df = recommender.train_on_playlist(playlist_id)
                    st.success("✅ Playlist analyzed successfully!")
                    st.session_state['recommender'] = recommender

        # Recommendation section
        if 'recommender' in st.session_state:
            st.header("2. Get Recommendations")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_song = st.selectbox(
                    "Choose a song:",
                    st.session_state.recommender.music_df['Track Name'].unique()
                )
            
            with col2:
                num_recommendations = st.slider(
                    "Number of recommendations",
                    min_value=1,
                    max_value=10,
                    value=5
                )
            
            with col3:
                approach = st.selectbox(
                    "Recommendation approach",
                    ["Hybrid", "Content-based", "Collaborative"]
                )
            
            if approach == "Hybrid":
                st.subheader("Weighting factors")
                col1, col2, col3 = st.columns(3)
                with col1:
                    content_weight = st.slider("Content-based weight", 0.0, 1.0, 0.4)
                with col2:
                    collab_weight = st.slider("Collaborative weight", 0.0, 1.0, 0.4)
                with col3:
                    popularity_weight = st.slider("Popularity weight", 0.0, 1.0, 0.2)
                
                # Normalize weights
                total = content_weight + collab_weight + popularity_weight
                content_weight /= total
                collab_weight /= total
                popularity_weight /= total
            
            if st.button("Get Recommendations"):
                with st.spinner("Finding similar songs..."):
                    if approach == "Hybrid":
                        recommendations = st.session_state.recommender.get_hybrid_recommendations(
                            selected_song,
                            num_recommendations=num_recommendations,
                            content_weight=content_weight,
                            collab_weight=collab_weight,
                            popularity_weight=popularity_weight
                        )
                    elif approach == "Content-based":
                        recommendations = st.session_state.recommender.get_content_based_recommendations(
                            selected_song,
                            num_recommendations=num_recommendations
                        )
                    else:  # Collaborative
                        recommendations = st.session_state.recommender.get_collaborative_recommendations(
                            selected_song,
                            num_recommendations=num_recommendations
                        )
                    
                    # Display recommendations
                    st.subheader("Recommended Songs")
                    
                    for idx, row in recommendations.iterrows():
                        with st.container():
                            col1, col2, col3 = st.columns([3, 2, 1])
                            with col1:
                                st.markdown(f"**{row['Track Name']}**")
                                st.write(f"by {row['Artists']}")
                                if row['Preview URL']:
                                    st.audio(row['Preview URL'])
                            with col2:
                                st.write(f"Album: {row['Album Name']}")
                                st.write(f"Released: {row['Release Date']}")
                            with col3:
                                st.write(f"Popularity: {row['Popularity']}")
                                if 'Hybrid Score' in row:
                                    st.write(f"Score: {row['Hybrid Score']:.3f}")
                                else:
                                    st.write(f"Similarity: {row['Similarity Score']:.3f}")
                                st.markdown(f"[Open in Spotify]({row['External URL']})")
                            st.divider()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your inputs and try again.")

if __name__ == "__main__":
    main()