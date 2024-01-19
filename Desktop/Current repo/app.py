from flask import Flask, render_template
import pandas as pd
import os

app = Flask(__name__, static_folder='static')

# Assume you have a function to fetch top listened data for each category

def fetch_top_listened_data(category, year):
    # Adjust file_path based on the selected year
    if year == 'all':
        file_path = f'data/top_listened_{category}_data.csv'
        df = pd.read_csv(file_path)
        # Convert DataFrame to a list of lists
        data_list = df.values.tolist()
        # Add column names as the first row
        header = df.columns.tolist()
        data_list.insert(0, header)
        return data_list
    else:
        file_path = f'data/top_listened_{category}_data_{year}.csv'
        df = pd.read_csv(file_path)
        # Convert DataFrame to a list of lists
        data_list = df.values.tolist()
        # Add column names as the first row
        header = df.columns.tolist()
        data_list.insert(0, header)
        return data_list



file_path = 'data/top_artists_weighted_2024.csv'
df = pd.read_csv(file_path)

data_list = df.values.tolist()
# Add column names as the first row
header = df.columns.tolist()
data_list.insert(0, header)

predictions_2024 = data_list

file_path = 'data/top_artists_weighted.csv'
df = pd.read_csv(file_path)

data_list = df.values.tolist()
# Add column names as the first row
header = df.columns.tolist()
data_list.insert(0, header)

predictions_overall = data_list




file_path = 'data/top_albums_weighted_2024.csv'
df = pd.read_csv(file_path)

data_list = df.values.tolist()
# Add column names as the first row
header = df.columns.tolist()
data_list.insert(0, header)

predictions_2024_album = data_list


file_path = 'data/top_albums_weighted.csv'
df = pd.read_csv(file_path)

data_list = df.values.tolist()
# Add column names as the first row
header = df.columns.tolist()
data_list.insert(0, header)

predictions_overall_album = data_list


file_path = 'data/top_songs_weighted_2024.csv'
df = pd.read_csv(file_path)

data_list = df.values.tolist()
# Add column names as the first row
header = df.columns.tolist()
data_list.insert(0, header)

predictions_2024_song = data_list

file_path = 'data/top_songs_weighted.csv'
df = pd.read_csv(file_path)

data_list = df.values.tolist()
# Add column names as the first row
header = df.columns.tolist()
data_list.insert(0, header)

predictions_overall_song = data_list



@app.route('/')
def index():
    return render_template('index.html')


#predictions

@app.route('/prediction')
def index_prediction():
    return render_template('prediction.html')


@app.route('/predictions_artists')
def predictions_artists():
    return render_template('prediction-artist.html', predictions_2024=predictions_2024, predictions_overall_data = predictions_overall)

@app.route('/predictions_albums')
def predictions_albums():
    return render_template('prediction-album.html', predictions_2024=predictions_2024_album, predictions_overall_data = predictions_overall_album)

@app.route('/predictions_songs')
def predictions_songs():
    return render_template('prediction-song.html', predictions_2024=predictions_2024_song, predictions_overall_data = predictions_overall_song)



#all


@app.route('/artists')
def artists():
    top_listened_artists = fetch_top_listened_data('artists', "all")
    return render_template('artists.html', top_listened_artists=top_listened_artists)

@app.route('/songs')
def songs():
    top_listened_songs = fetch_top_listened_data('songs', "all")
    return render_template('songs.html', top_listened_songs=top_listened_songs)

@app.route('/albums')
def albums():
    top_listened_albums = fetch_top_listened_data('albums', "all")
    return render_template('albums.html', top_listened_albums=top_listened_albums)

#2021

@app.route('/2021')
def index2021():
    return render_template('index-2021.html')

@app.route('/artists-2021')
def artists2021():
    top_listened_artists = fetch_top_listened_data('artists', "2021")
    return render_template('artists-2021.html', top_listened_artists=top_listened_artists)

@app.route('/songs-2021')
def songs2021():
    top_listened_songs = fetch_top_listened_data('songs', "2021")
    return render_template('songs-2021.html', top_listened_songs=top_listened_songs)

@app.route('/albums-2021')
def albums2021():
    top_listened_albums = fetch_top_listened_data('albums', "2021")
    return render_template('albums-2021.html', top_listened_albums=top_listened_albums)

#2022

@app.route('/2022')
def index2022():
    return render_template('index-2022.html')

@app.route('/artists-2022')
def artists2022():
    top_listened_artists = fetch_top_listened_data('artists', "2022")
    return render_template('artists-2022.html', top_listened_artists=top_listened_artists)

@app.route('/songs-2022')
def songs2022():
    top_listened_songs = fetch_top_listened_data('songs', "2022")
    return render_template('songs-2022.html', top_listened_songs=top_listened_songs)

@app.route('/albums-2022')
def albums2022():
    top_listened_albums = fetch_top_listened_data('albums', "2022")
    return render_template('albums-2022.html', top_listened_albums=top_listened_albums)

#2023

@app.route('/2023')
def index2023():
    return render_template('index-2023.html')


@app.route('/artists-2023')
def artists2023():
    top_listened_artists = fetch_top_listened_data('artists', "2023")
    return render_template('artists-2023.html', top_listened_artists=top_listened_artists)

@app.route('/songs-2023')
def songs2023():
    top_listened_songs = fetch_top_listened_data('songs', "2023")
    return render_template('songs-2023.html', top_listened_songs=top_listened_songs)

@app.route('/albums-2023')
def albums2023():
    top_listened_albums = fetch_top_listened_data('albums', "2023")
    return render_template('albums-2023.html', top_listened_albums=top_listened_albums)



if __name__ == '__main__':
    app.run(debug=True)
