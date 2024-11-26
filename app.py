from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the dataset (ensure the path to the CSV is correct)
df = pd.read_csv('Zomato data.csv')

# Function to predict restaurant type based on inputs
def predict_restaurant_type(restaurant_type, approx_cost, online_order, votes, location, cuisines, timings, rating):
    # Example logic: refine prediction based on inputs
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
    elif location in df['location'].values:
        return f"Restaurant in {location} with good ratings"
    else:
        return "Generic Restaurant"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    restaurant_type = request.form.get('restaurant_type')
    approx_cost = float(request.form.get('approx_cost'))
    rating = float(request.form.get('rating'))
    votes = int(request.form.get('votes'))
    online_order = request.form.get('online_order')
    location = request.form.get('location')
    cuisines = request.form.get('cuisines')
    timings = request.form.get('timings')

    # Predict the restaurant type based on input data
    prediction = predict_restaurant_type(
        restaurant_type, approx_cost, online_order, votes, location, cuisines, timings, rating)

    # Return the prediction result
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
