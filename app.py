from flask import Flask, render_template
import pandas as pd
import os

app = Flask(__name__, static_folder='static')

# Assume you have a function to fetch top listened data for each category
def fetch_top_listened_data(category):
    file_path = f'top_listened_{category}_data.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Convert DataFrame to a list of lists
        data_list = df.values.tolist()
        # Add column names as the first row
        header = df.columns.tolist()
        data_list.insert(0, header)
        return data_list
    else:
        return []
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/artists')
def artists():
    top_listened_artists = fetch_top_listened_data('artists')
    return render_template('artists.html', top_listened_artists=top_listened_artists)

@app.route('/songs')
def songs():
    top_listened_songs = fetch_top_listened_data('songs')
    return render_template('songs.html', top_listened_songs=top_listened_songs)

@app.route('/albums')
def albums():
    top_listened_albums = fetch_top_listened_data('albums')
    return render_template('albums.html', top_listened_albums=top_listened_albums)

if __name__ == '__main__':
    app.run(debug=True)
