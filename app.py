from flask import Flask, render_template
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def index():
    # Assuming you have CSV files named 'top_listened_artists_data.csv', 'top_listened_songs_data.csv', etc.
    files = ['top_listened_artists_data.csv', 'top_listened_songs_data.csv', 'top_listened_genres_data.csv', 'top_listened_albums_data.csv']
    data = {}

    for file in files:
        df = pd.read_csv(file)
        data[os.path.splitext(file)[0]] = df.to_html(index=False)

    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
