# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
from utils import odi_model

# Load the Random Forest CLassifier model
filename = 'model/first-innings-score-lr-model.pkl'
regressor = pickle.load(open(filename, 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    title = "First Innings Score Predictor for "
    match_type = "Indian Premier League (IPL)"
    return render_template('index.html', title=title, match_type=match_type)


@app.route('/predict', methods=['POST'])
def predict():
    title = "First Innings Score Predictor for "
    match_type = "Indian Premier League (IPL)"
    temp_array = list()

    if request.method == 'POST':

        batting_team = request.form['batting-team']
        if batting_team == 'Chennai Super Kings':
            temp_array = temp_array + [1, 0, 0, 0, 0, 0, 0, 0]
        elif batting_team == 'Delhi Daredevils':
            temp_array = temp_array + [0, 1, 0, 0, 0, 0, 0, 0]
        elif batting_team == 'Kings XI Punjab':
            temp_array = temp_array + [0, 0, 1, 0, 0, 0, 0, 0]
        elif batting_team == 'Kolkata Knight Riders':
            temp_array = temp_array + [0, 0, 0, 1, 0, 0, 0, 0]
        elif batting_team == 'Mumbai Indians':
            temp_array = temp_array + [0, 0, 0, 0, 1, 0, 0, 0]
        elif batting_team == 'Rajasthan Royals':
            temp_array = temp_array + [0, 0, 0, 0, 0, 1, 0, 0]
        elif batting_team == 'Royal Challengers Bangalore':
            temp_array = temp_array + [0, 0, 0, 0, 0, 0, 1, 0]
        elif batting_team == 'Sunrisers Hyderabad':
            temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 1]

        bowling_team = request.form['bowling-team']
        if bowling_team == 'Chennai Super Kings':
            temp_array = temp_array + [1, 0, 0, 0, 0, 0, 0, 0]
        elif bowling_team == 'Delhi Daredevils':
            temp_array = temp_array + [0, 1, 0, 0, 0, 0, 0, 0]
        elif bowling_team == 'Kings XI Punjab':
            temp_array = temp_array + [0, 0, 1, 0, 0, 0, 0, 0]
        elif bowling_team == 'Kolkata Knight Riders':
            temp_array = temp_array + [0, 0, 0, 1, 0, 0, 0, 0]
        elif bowling_team == 'Mumbai Indians':
            temp_array = temp_array + [0, 0, 0, 0, 1, 0, 0, 0]
        elif bowling_team == 'Rajasthan Royals':
            temp_array = temp_array + [0, 0, 0, 0, 0, 1, 0, 0]
        elif bowling_team == 'Royal Challengers Bangalore':
            temp_array = temp_array + [0, 0, 0, 0, 0, 0, 1, 0]
        elif bowling_team == 'Sunrisers Hyderabad':
            temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 1]

        overs = float(request.form['overs'])
        runs = int(request.form['runs'])
        wickets = int(request.form['wickets'])
        runs_in_prev_5 = int(request.form['runs_in_prev_5'])
        wickets_in_prev_5 = int(request.form['wickets_in_prev_5'])

        temp_array = temp_array + [overs, runs,
                                   wickets, runs_in_prev_5, wickets_in_prev_5]

        data_ipl = np.array([temp_array])
        my_prediction = int(regressor.predict(data_ipl)[0])

        return render_template('result.html', lower_limit=my_prediction-10, upper_limit=my_prediction+5, title=title, match_type=match_type)


@app.route('/odi')
def odi():
    title = "Match Winner Predictor "
    match_type = "One Day International (ODI)"
    return render_template('odi.html', title=title, match_type=match_type)


@app.route('/odi-predict', methods=['POST'])
def odi_prediction():
    title = "Match Winner Predictor "
    match_type = "One Day International (ODI)"
    if request.method == 'POST':

        runs = int(request.form.get('runs'))
        wickets = int(request.form.get('wickets'))
        overs = float(request.form.get('overs'))
        striker = int(request.form.get('striker'))
        non_striker = int(request.form.get('non-striker'))
        total = int(request.form.get('target'))

    data = np.array([[runs, wickets, overs, striker, non_striker, total]])
    prediction = odi_model.predict(data)
    prediction = "{0:.2f}".format(prediction)

    return render_template('odi-result.html', prediction=prediction, title=title, match_type=match_type)


if __name__ == '__main__':
    app.run(debug=True)
