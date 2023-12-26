import pandas as pd
import glob

json_files = glob.glob('*.json')

# Initialize an empty DataFrame to store the combined data
combined_df = pd.DataFrame()

# Loop through each JSON file and concatenate the data
for file in json_files:
    df = pd.read_json(file)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# Convert 'ts' column to datetime format
combined_df['ts'] = pd.to_datetime(combined_df['ts'])

# Calculate listening time for each artist, song, genre, album
listening_time_artists = combined_df.groupby('master_metadata_album_artist_name')['ms_played'].sum() / (60 * 1000)
listening_time_songs = combined_df.groupby('master_metadata_track_name')['ms_played'].sum() / (60 * 1000)
listening_time_genres = combined_df.groupby('master_metadata_album_artist_name')['ms_played'].sum() / (60 * 1000)
listening_time_albums = combined_df.groupby('master_metadata_album_album_name')['ms_played'].sum() / (60 * 1000)

# Get the top 10 listening times for each category
top_listened_artists = listening_time_artists.nlargest(10)
top_listened_songs = listening_time_songs.nlargest(10)
top_listened_genres = listening_time_genres.nlargest(10)
top_listened_albums = listening_time_albums.nlargest(10)

# Create DataFrames for the top listened data
top_listened_artists_df = pd.DataFrame({
    'Top Listened Artists': top_listened_artists.index,
    'Listening Time (minutes)': top_listened_artists.values
})

top_listened_songs_df = pd.DataFrame({
    'Top Listened Songs': top_listened_songs.index,
    'Listening Time (minutes)': top_listened_songs.values
})

top_listened_genres_df = pd.DataFrame({
    'Top Listened Genres': top_listened_genres.index,
    'Listening Time (minutes)': top_listened_genres.values
})

top_listened_albums_df = pd.DataFrame({
    'Top Listened Albums': top_listened_albums.index,
    'Listening Time (minutes)': top_listened_albums.values
})

# Save the DataFrames to CSV files
top_listened_artists_df.to_csv('top_listened_artists_data.csv', index=False)
top_listened_songs_df.to_csv('top_listened_songs_data.csv', index=False)
top_listened_genres_df.to_csv('top_listened_genres_data.csv', index=False)
top_listened_albums_df.to_csv('top_listened_albums_data.csv', index=False)

# Print or use the information as needed
print(f'Top 10 Listened Artists:\n{top_listened_artists}')
print(f'Top 10 Listened Songs:\n{top_listened_songs}')
print(f'Top 10 Listened Genres:\n{top_listened_genres}')
print(f'Top 10 Listened Albums:\n{top_listened_albums}')
