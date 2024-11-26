from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('Zomato data.csv')

# Function to predict restaurant type based on inputs
def predict_restaurant_type(restaurant_type, approx_cost, online_order, votes, location, cuisines, timings, rating):
    # Example simple prediction logic based on user input
    if restaurant_type == 'Casual Dining' and approx_cost > 500:
        return "High-end Casual Dining"
    elif restaurant_type == 'Cafe' and approx_cost <= 500:
        return "Budget-friendly Cafe"
    elif online_order == 'Yes':
        return "Online Order Preferred"
    elif rating >= 4:
        return "Top Rated Restaurant"
    elif votes > 1000:
        return "Popular Restaurant"
    else:
        return "Generic Restaurant"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    restaurant_type = request.form.get('feature1')
    approx_cost = float(request.form.get('feature2'))
    rating = float(request.form.get('feature6'))
    votes = int(request.form.get('feature4'))
    online_order = request.form.get('feature3')
    location = request.form.get('feature5')
    cuisines = request.form.get('feature7')
    timings = request.form.get('feature8')

    # Use the input data to predict the restaurant type or other details
    prediction = predict_restaurant_type(restaurant_type, approx_cost, online_order, votes, location, cuisines, timings, rating)

    # Return the prediction result
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
