import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.linear_model import LinearRegression

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
top_listened_artists = listening_time_artists.nlargest(10).reset_index()
top_listened_songs = listening_time_songs.nlargest(10).reset_index()
top_listened_genres = listening_time_genres.nlargest(10).reset_index()
top_listened_albums = listening_time_albums.nlargest(10).reset_index()

# Round the values to two decimal places
top_listened_artists['ms_played'] = top_listened_artists['ms_played'].round(2)
top_listened_songs['ms_played'] = top_listened_songs['ms_played'].round(2)
top_listened_genres['ms_played'] = top_listened_genres['ms_played'].round(2)
top_listened_albums['ms_played'] = top_listened_albums['ms_played'].round(2)

# Create DataFrames for the top listened data
top_listened_artists_df = pd.DataFrame({
    'Top Listened Artists': top_listened_artists['master_metadata_album_artist_name'],
    'Listening Time (minutes)': top_listened_artists['ms_played']
})

top_listened_songs_df = pd.DataFrame({
    'Top Listened Songs': top_listened_songs['master_metadata_track_name'],
    'Listening Time (minutes)': top_listened_songs['ms_played']
})


top_listened_albums_df = pd.DataFrame({
    'Top Listened Albums': top_listened_albums['master_metadata_album_album_name'],
    'Listening Time (minutes)': top_listened_albums['ms_played']
})

# Save the DataFrames to CSV files
top_listened_artists_df.to_csv('top_listened_artists_data.csv', index=False)
top_listened_songs_df.to_csv('top_listened_songs_data.csv', index=False)
top_listened_albums_df.to_csv('top_listened_albums_data.csv', index=False)

# Print or use the information as needed
print(f'Top 10 Listened Artists:\n{top_listened_artists}')
print(f'Top 10 Listened Songs:\n{top_listened_songs}')
print(f'Top 10 Listened Albums:\n{top_listened_albums}')






#2021





json_files = ["1.json"]

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
top_listened_artists = listening_time_artists.nlargest(10).reset_index()
top_listened_songs = listening_time_songs.nlargest(10).reset_index()
top_listened_genres = listening_time_genres.nlargest(10).reset_index()
top_listened_albums = listening_time_albums.nlargest(10).reset_index()

# Round the values to two decimal places
top_listened_artists['ms_played'] = top_listened_artists['ms_played'].round(2)
top_listened_songs['ms_played'] = top_listened_songs['ms_played'].round(2)
top_listened_genres['ms_played'] = top_listened_genres['ms_played'].round(2)
top_listened_albums['ms_played'] = top_listened_albums['ms_played'].round(2)

# Create DataFrames for the top listened data
top_listened_artists_df = pd.DataFrame({
    'Top Listened Artists': top_listened_artists['master_metadata_album_artist_name'],
    'Listening Time (minutes)': top_listened_artists['ms_played']
})

top_listened_songs_df = pd.DataFrame({
    'Top Listened Songs': top_listened_songs['master_metadata_track_name'],
    'Listening Time (minutes)': top_listened_songs['ms_played']
})


top_listened_albums_df = pd.DataFrame({
    'Top Listened Albums': top_listened_albums['master_metadata_album_album_name'],
    'Listening Time (minutes)': top_listened_albums['ms_played']
})

# Save the DataFrames to CSV files
top_listened_artists_df.to_csv('top_listened_artists_data_2021.csv', index=False)
top_listened_songs_df.to_csv('top_listened_songs_data_2021.csv', index=False)
top_listened_albums_df.to_csv('top_listened_albums_data_2021.csv', index=False)

# Print or use the information as needed
print(f'Top 10 Listened Artists:\n{top_listened_artists}')
print(f'Top 10 Listened Songs:\n{top_listened_songs}')
print(f'Top 10 Listened Albums:\n{top_listened_albums}')





#2022 




json_files = ["3.json"]

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
top_listened_artists = listening_time_artists.nlargest(10).reset_index()
top_listened_songs = listening_time_songs.nlargest(10).reset_index()
top_listened_genres = listening_time_genres.nlargest(10).reset_index()
top_listened_albums = listening_time_albums.nlargest(10).reset_index()

# Round the values to two decimal places
top_listened_artists['ms_played'] = top_listened_artists['ms_played'].round(2)
top_listened_songs['ms_played'] = top_listened_songs['ms_played'].round(2)
top_listened_genres['ms_played'] = top_listened_genres['ms_played'].round(2)
top_listened_albums['ms_played'] = top_listened_albums['ms_played'].round(2)

# Create DataFrames for the top listened data
top_listened_artists_df = pd.DataFrame({
    'Top Listened Artists': top_listened_artists['master_metadata_album_artist_name'],
    'Listening Time (minutes)': top_listened_artists['ms_played']
})

top_listened_songs_df = pd.DataFrame({
    'Top Listened Songs': top_listened_songs['master_metadata_track_name'],
    'Listening Time (minutes)': top_listened_songs['ms_played']
})


top_listened_albums_df = pd.DataFrame({
    'Top Listened Albums': top_listened_albums['master_metadata_album_album_name'],
    'Listening Time (minutes)': top_listened_albums['ms_played']
})

# Save the DataFrames to CSV files
top_listened_artists_df.to_csv('top_listened_artists_data_2022.csv', index=False)
top_listened_songs_df.to_csv('top_listened_songs_data_2022.csv', index=False)
top_listened_albums_df.to_csv('top_listened_albums_data_2022.csv', index=False)

# Print or use the information as needed
print(f'Top 10 Listened Artists:\n{top_listened_artists}')
print(f'Top 10 Listened Songs:\n{top_listened_songs}')
print(f'Top 10 Listened Albums:\n{top_listened_albums}')




#2023 



json_files = ["5.json"]

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
top_listened_artists = listening_time_artists.nlargest(10).reset_index()
top_listened_songs = listening_time_songs.nlargest(10).reset_index()
top_listened_genres = listening_time_genres.nlargest(10).reset_index()
top_listened_albums = listening_time_albums.nlargest(10).reset_index()

# Round the values to two decimal places
top_listened_artists['ms_played'] = top_listened_artists['ms_played'].round(2)
top_listened_songs['ms_played'] = top_listened_songs['ms_played'].round(2)
top_listened_genres['ms_played'] = top_listened_genres['ms_played'].round(2)
top_listened_albums['ms_played'] = top_listened_albums['ms_played'].round(2)

# Create DataFrames for the top listened data
top_listened_artists_df = pd.DataFrame({
    'Top Listened Artists': top_listened_artists['master_metadata_album_artist_name'],
    'Listening Time (minutes)': top_listened_artists['ms_played']
})

top_listened_songs_df = pd.DataFrame({
    'Top Listened Songs': top_listened_songs['master_metadata_track_name'],
    'Listening Time (minutes)': top_listened_songs['ms_played']
})


top_listened_albums_df = pd.DataFrame({
    'Top Listened Albums': top_listened_albums['master_metadata_album_album_name'],
    'Listening Time (minutes)': top_listened_albums['ms_played']
})

# Save the DataFrames to CSV files
top_listened_artists_df.to_csv('top_listened_artists_data_2023.csv', index=False)
top_listened_songs_df.to_csv('top_listened_songs_data_2023.csv', index=False)
top_listened_albums_df.to_csv('top_listened_albums_data_2023.csv', index=False)

# Print or use the information as needed
print(f'Top 10 Listened Artists:\n{top_listened_artists}')
print(f'Top 10 Listened Songs:\n{top_listened_songs}')
print(f'Top 10 Listened Albums:\n{top_listened_albums}')


















import pandas as pd
import json

# Read 2021 data from 1.json
with open('1.json', 'r', encoding='utf-8') as f:
    data_2021 = json.load(f)

# Read 2022 data from 3.json
with open('3.json', 'r', encoding='utf-8') as f:
    data_2022 = json.load(f)

# Read 2023 data from 5.json
with open('5.json', 'r', encoding='utf-8') as f:
    data_2023 = json.load(f)

# Convert JSON data to DataFrames
df_2021 = pd.json_normalize(data_2021)
df_2022 = pd.json_normalize(data_2022)
df_2023 = pd.json_normalize(data_2023)

# Convert milliseconds to minutes
df_2021['minutes_played'] = df_2021['ms_played'] / (1000 * 60)
df_2022['minutes_played'] = df_2022['ms_played'] / (1000 * 60)
df_2023['minutes_played'] = df_2023['ms_played'] / (1000 * 60)

# Extract the year from the timestamp
df_2021['year'] = pd.to_datetime(df_2021['ts']).dt.year
df_2022['year'] = pd.to_datetime(df_2022['ts']).dt.year
df_2023['year'] = pd.to_datetime(df_2023['ts']).dt.year
weights = {2021: 0.2, 2022: 0.5, 2023: 1.0}

# Calculate the total playtime for each artist considering the weights
df_2021['weighted_playtime'] = df_2021['minutes_played'] * df_2021['year'].map(lambda x: weights.get(x, 0))
df_2022['weighted_playtime'] = df_2022['minutes_played'] * df_2022['year'].map(lambda x: weights.get(x, 0))
df_2023['weighted_playtime'] = df_2023['minutes_played'] * df_2023['year'].map(lambda x: weights.get(x, 0))

# Concatenate the dataframes
df = pd.concat([df_2021, df_2022, df_2023], ignore_index=True)

# Group by artist and calculate the sum of weighted playtime
weighted_artist_playtime = df.groupby('master_metadata_album_artist_name')['weighted_playtime'].sum()

# Identify the top 10 most likely artists with the highest sum of weighted playtime
top_artists_weighted = weighted_artist_playtime.nlargest(10).round(2)

# Save the results to a CSV file
top_artists_weighted.to_csv('top_artists_weighted.csv', header=True)

# Combine 2021 and 2022 data for prediction
df_train = pd.concat([df_2021, df_2022])

# Calculate the total playtime for each artist in a specific year
df_train['weighted_playtime'] = df_train.groupby('master_metadata_album_artist_name')['minutes_played'].transform('sum') * df_train['year']

# Group by artist and calculate the sum of weighted playtime for each year
weighted_artist_playtime_train = df_train.groupby('master_metadata_album_artist_name')['weighted_playtime'].sum()
weights = {2021: 0.2, 2022: 0.5, 2023: 1.0}

# Predict likely artists for 2024
df_2024 = pd.read_json("6.json")  # Replace with the actual data for 2024
df_2024['minutes_played'] = df_2024['ms_played'] / (1000 * 60)
df_2024['year'] = pd.to_datetime(df_2024['ts']).dt.year
df_2024['weighted_playtime'] = df_2024.groupby('master_metadata_album_artist_name')['minutes_played'].transform('sum') * df_2024['year'].map(weights)
weighted_artist_playtime_2024 = df_2024.groupby('master_metadata_album_artist_name')['weighted_playtime'].sum()
top_artists_weighted_2024 = weighted_artist_playtime_2024.nlargest(10).round(2)

# Save the results for 2024 to a CSV file
top_artists_weighted_2024.to_csv('top_artists_weighted_2024.csv', header=True)

# Display the top 10 most likely artists for each year
print("Top 10 Most Likely Artists Overall:")
print(top_artists_weighted)
print("\nTop 10 Most Likely Artists for 2024:")
print(top_artists_weighted_2024)






# Similar process for top albums and songs

# Group by album and calculate the sum of weighted playtime
weighted_album_playtime = df.groupby('master_metadata_album_album_name')['weighted_playtime'].sum()

# Identify the top 10 most likely albums with the highest sum of weighted playtime
top_albums_weighted = weighted_album_playtime.nlargest(10).round(2)

# Save the results to a CSV file
top_albums_weighted.to_csv('top_albums_weighted.csv', header=True)

# Combine 2021 and 2022 data for prediction
df_train_albums = pd.concat([df_2021, df_2022])

# Calculate the total playtime for each album in a specific year
df_train_albums['weighted_playtime'] = df_train_albums.groupby('master_metadata_album_album_name')['minutes_played'].transform('sum') * df_train_albums['year']

# Group by album and calculate the sum of weighted playtime for each year
weighted_album_playtime_train = df_train_albums.groupby('master_metadata_album_album_name')['weighted_playtime'].sum()
weights = {2021: 0.2, 2022: 0.5, 2023: 1.0}

# Predict likely albums for 2024
df_2024_albums = pd.read_json("6.json")  # Replace with the actual data for 2024
df_2024_albums['minutes_played'] = df_2024_albums['ms_played'] / (1000 * 60)
df_2024_albums['year'] = pd.to_datetime(df_2024_albums['ts']).dt.year
df_2024_albums['weighted_playtime'] = df_2024_albums.groupby('master_metadata_album_album_name')['minutes_played'].transform('sum') * df_2024_albums['year'].map(weights)
weighted_album_playtime_2024 = df_2024_albums.groupby('master_metadata_album_album_name')['weighted_playtime'].sum()
top_albums_weighted_2024 = weighted_album_playtime_2024.nlargest(10).round(2)

# Save the results for 2024 to a CSV file
top_albums_weighted_2024.to_csv('top_albums_weighted_2024.csv', header=True)

# Display the top 10 most likely albums for each year
print("\nTop 10 Most Likely Albums Overall:")
print(top_albums_weighted)
print("\nTop 10 Most Likely Albums for 2024:")
print(top_albums_weighted_2024)


# Group by song and calculate the sum of weighted playtime
weighted_song_playtime = df.groupby('master_metadata_track_name')['weighted_playtime'].sum()

# Identify the top 10 most likely songs with the highest sum of weighted playtime
top_songs_weighted = weighted_song_playtime.nlargest(10).round(2)

# Save the results to a CSV file
top_songs_weighted.to_csv('top_songs_weighted.csv', header=True)

# Combine 2021 and 2022 data for prediction
df_train_songs = pd.concat([df_2021, df_2022])

# Calculate the total playtime for each song in a specific year
df_train_songs['weighted_playtime'] = df_train_songs.groupby('master_metadata_track_name')['minutes_played'].transform('sum') * df_train_songs['year']

# Group by song and calculate the sum of weighted playtime for each year
weighted_song_playtime_train = df_train_songs.groupby('master_metadata_track_name')['weighted_playtime'].sum()
weights = {2021: 0.2, 2022: 0.5, 2023: 1.0}

# Predict likely songs for 2024
df_2024_songs = pd.read_json("6.json")  # Replace with the actual data for 2024
df_2024_songs['minutes_played'] = df_2024_songs['ms_played'] / (1000 * 60)
df_2024_songs['year'] = pd.to_datetime(df_2024_songs['ts']).dt.year
df_2024_songs['weighted_playtime'] = df_2024_songs.groupby('master_metadata_track_name')['minutes_played'].transform('sum') * df_2024_songs['year'].map(weights)
weighted_song_playtime_2024 = df_2024_songs.groupby('master_metadata_track_name')['weighted_playtime'].sum()
top_songs_weighted_2024 = weighted_song_playtime_2024.nlargest(10).round(2)

# Save the results for 2024 to a CSV file
top_songs_weighted_2024.to_csv('top_songs_weighted_2024.csv', header=True)

# Display the top 10 most likely songs for each year
print("\nTop 10 Most Likely Songs Overall:")
print(top_songs_weighted)
print("\nTop 10 Most Likely Songs for 2024:")
print(top_songs_weighted_2024)










import matplotlib.pyplot as plt
import pandas as pd

# Function to create a bar chart
def create_bar_chart(ax, data, title):
    data.plot(kind='bar', x=0, y=1, ax=ax, legend=False)
    ax.set_title(title)
    ax.set_ylabel('Listening Time (minutes)')
    ax.set_xlabel('Top Listened')
    ax.tick_params(axis='x', rotation=45)

# Read the top listened data from CSV files
top_artists_2021 = pd.read_csv('top_listened_artists_data_2021.csv')
top_songs_2021 = pd.read_csv('top_listened_songs_data_2021.csv')
top_albums_2021 = pd.read_csv('top_listened_albums_data_2021.csv')

top_artists_2022 = pd.read_csv('top_listened_artists_data_2022.csv')
top_songs_2022 = pd.read_csv('top_listened_songs_data_2022.csv')
top_albums_2022 = pd.read_csv('top_listened_albums_data_2022.csv')

top_artists_2023 = pd.read_csv('top_listened_artists_data_2023.csv')
top_songs_2023 = pd.read_csv('top_listened_songs_data_2023.csv')
top_albums_2023 = pd.read_csv('top_listened_albums_data_2023.csv')

# Combine data for each year
top_artists_list = [top_artists_2021, top_artists_2022, top_artists_2023]
top_songs_list = [top_songs_2021, top_songs_2022, top_songs_2023]
top_albums_list = [top_albums_2021, top_albums_2022, top_albums_2023]

# Create subplots for each year
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
fig.suptitle('Top 10 Listened Artists, Songs, and Albums')

# Iterate through each year and create bar charts
for i, (artists, songs, albums) in enumerate(zip(top_artists_list, top_songs_list, top_albums_list), start=2021):
    create_bar_chart(axes[i-2021, 0], artists, f'Top 10 Listened Artists - {i}')
    create_bar_chart(axes[i-2021, 1], songs, f'Top 10 Listened Songs - {i}')
    create_bar_chart(axes[i-2021, 2], albums, f'Top 10 Listened Albums - {i}')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout
plt.show()












import matplotlib.pyplot as plt
import pandas as pd

# Function to create a horizontal bar chart and save as an image
def create_and_save_horizontal_bar_chart_song(data, title, filename):
    fig, ax = plt.subplots(figsize=(25, 20))  # Increase the figure size
    data.sort_values(by='Listening Time (minutes)', ascending=True).plot(kind='barh', x='Top Listened Songs', y='Listening Time (minutes)', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Listening Time (minutes)')
    ax.set_ylabel('Top Listened Songs')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def create_and_save_horizontal_bar_chart_album(data, title, filename):
    fig, ax = plt.subplots(figsize=(25, 20))  # Increase the figure size
    data.sort_values(by='Listening Time (minutes)', ascending=True).plot(kind='barh', x='Top Listened Albums', y='Listening Time (minutes)', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Listening Time (minutes)')
    ax.set_ylabel('Top Listened Albums')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



def create_and_save_horizontal_bar_chart_artist(data, title, filename):
    fig, ax = plt.subplots(figsize=(8, 10))  # Increase the figure size
    data.sort_values(by='Listening Time (minutes)', ascending=True).plot(kind='barh', x='Top Listened Artists', y='Listening Time (minutes)', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Listening Time (minutes)')
    ax.set_ylabel('Top Listened Artists')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()





def create_and_save_horizontal_bar_chart_ml_album(data, title, filename):
    fig, ax = plt.subplots(figsize=(8, 10))  # Increase the figure size
    data.sort_values(by='master_metadata_album_album_name', ascending=True).plot(kind='barh', x='weighted_playtime', y='master_metadata_album_album_name', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('master_metadata_album_album_name')
    ax.set_ylabel('weighted_playtime')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def create_and_save_horizontal_bar_chart_ml_artist(data, title, filename):
    fig, ax = plt.subplots(figsize=(8, 10))  # Increase the figure size
    data.sort_values(by='master_metadata_album_artist_name', ascending=True).plot(kind='barh', x='master_metadata_album_artist_name', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('master_metadata_album_artist_name')
    ax.set_ylabel('weighted_playtime')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def create_and_save_horizontal_bar_chart_ml_song(data, title, filename):
    fig, ax = plt.subplots(figsize=(8, 10))  # Increase the figure size
    data.sort_values(by='master_metadata_track_name', ascending=True).plot(kind='barh', x='weighted_playtime', y='master_metadata_track_name', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('master_metadata_track_name')
    ax.set_ylabel('weighted_playtime')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



# Read the top listened data from CSV files
top_artists_2021 = pd.read_csv('top_listened_artists_data_2021.csv')
top_songs_2021 = pd.read_csv('top_listened_songs_data_2021.csv')
top_albums_2021 = pd.read_csv('top_listened_albums_data_2021.csv')

top_artists_2022 = pd.read_csv('top_listened_artists_data_2022.csv')
top_songs_2022 = pd.read_csv('top_listened_songs_data_2022.csv')
top_albums_2022 = pd.read_csv('top_listened_albums_data_2022.csv')

top_artists_2023 = pd.read_csv('top_listened_artists_data_2023.csv')
top_songs_2023 = pd.read_csv('top_listened_songs_data_2023.csv')
top_albums_2023 = pd.read_csv('top_listened_albums_data_2023.csv')





top_artists_overall = pd.read_csv('data/top_listened_artists_data.csv')
top_songs_overall = pd.read_csv('data/top_listened_songs_data.csv')
top_albums_overall = pd.read_csv('data/top_listened_albums_data.csv')

create_and_save_horizontal_bar_chart_artist(top_artists_overall, 'Top 10 Listened Artists - Overall', 'top_artists_overall.png')
create_and_save_horizontal_bar_chart_song(top_songs_overall, 'Top 10 Listened Songs - Overall', 'top_songs_overall.png')
create_and_save_horizontal_bar_chart_album(top_albums_overall, 'Top 10 Listened Albums - Overall', 'top_albums_overall.png')

top_artists_weighted= pd.read_csv('data/top_artists_weighted.csv')
top_songs_weighted= pd.read_csv('data/top_songs_weighted.csv')
top_albums_weighted = pd.read_csv('data/top_albums_weighted.csv')

create_and_save_horizontal_bar_chart_ml_artist(top_artists_weighted, 'Top 10 Listened Artists - Weighted', 'top_artists_weighted.png')
create_and_save_horizontal_bar_chart_ml_song(top_songs_weighted, 'Top 10 Listened Songs - Weighted', 'top_songs_weighted.png')
create_and_save_horizontal_bar_chart_ml_album(top_albums_weighted, 'Top 10 Listened Albums - Weighted', 'top_albums_weighted.png')


top_artists_ML= pd.read_csv('data/top_artists_weighted_2024.csv')
top_songs_ML = pd.read_csv('data/top_songs_weighted_2024.csv')
top_albums_ML = pd.read_csv('data/top_albums_weighted_2024.csv')

create_and_save_horizontal_bar_chart_ml(top_artists_ML, 'Top 10 Listened Artists - ML', 'top_artists_ML.png')
create_and_save_horizontal_bar_chart_ml(top_songs_ML, 'Top 10 Listened Songs - ML', 'top_songs_ML.png')
create_and_save_horizontal_bar_chart_ml(top_albums_ML, 'Top 10 Listened Albums - ML', 'top_albums_ML.png')



# Create and save horizontal bar charts for each category
create_and_save_horizontal_bar_chart_artist(top_artists_2021, 'Top 10 Listened Artists - 2021', 'top_artists_2021.png')
create_and_save_horizontal_bar_chart_song(top_songs_2021, 'Top 10 Listened Songs - 2021', 'top_songs_2021.png')
create_and_save_horizontal_bar_chart_album(top_albums_2021, 'Top 10 Listened Albums - 2021', 'top_albums_2021.png')

create_and_save_horizontal_bar_chart_artist(top_artists_2022, 'Top 10 Listened Artists - 2022', 'top_artists_2022.png')
create_and_save_horizontal_bar_chart_song(top_songs_2022, 'Top 10 Listened Songs - 2022', 'top_songs_2022.png')
create_and_save_horizontal_bar_chart_album(top_albums_2022, 'Top 10 Listened Albums - 2022', 'top_albums_2022.png')

create_and_save_horizontal_bar_chart_artist(top_artists_2023, 'Top 10 Listened Artists - 2023', 'top_artists_2023.png')
create_and_save_horizontal_bar_chart_song(top_songs_2023, 'Top 10 Listened Songs - 2023', 'top_songs_2023.png')
create_and_save_horizontal_bar_chart_album(top_albums_2023, 'Top 10 Listened Albums - 2023', 'top_albums_2023.png')
