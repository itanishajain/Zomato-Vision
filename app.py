from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

# Example function to simulate a prediction (you can replace it with your own model)
def predict_customer_satisfaction(data):
    # Dummy logic for predicting customer satisfaction (replace with your actual ML model)
    if float(data['rating']) > 4 and int(data['votes']) > 1000:
        return "High Satisfaction"
    elif float(data['rating']) < 3:
        return "Low Satisfaction"
    else:
        return "Moderate Satisfaction"

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract user inputs
    restaurant_type = request.form['restaurant_type']
    approx_cost = request.form['approx_cost']
    rating = request.form['rating']
    votes = request.form['votes']
    online_order = request.form['online_order']
    location = request.form['location']
    cuisines = request.form['cuisines']
    timings = request.form['timings']
    
    # Example of processing the data into a format for prediction
    data = {
        'restaurant_type': restaurant_type,
        'approx_cost': approx_cost,
        'rating': rating,
        'votes': votes,
        'online_order': online_order,
        'location': location,
        'cuisines': cuisines,
        'timings': timings
    }
    
    # Here you would load the actual model and make a prediction (use model.predict())
    prediction = predict_customer_satisfaction(data)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
