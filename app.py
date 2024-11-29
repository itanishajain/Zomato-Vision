from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask App
app = Flask(__name__)

# Load the pre-trained model (you should train and save it beforehand)
model = pickle.load(open('zomato_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    name = request.form['name']
    online_order = 1 if request.form['online_order'] == 'Yes' else 0
    book_table = 1 if request.form['book_table'] == 'Yes' else 0
    rate = float(request.form['rate'].split('/')[0])
    votes = int(request.form['votes'])
    approx_cost = int(request.form['approx_cost'])
    listed_in = request.form['listed_in']
    listed_in_encoded = {'Buffet': 0, 'Cafes': 1, 'Dining': 2, 'Other': 3}.get(listed_in, 0)

    # Prepare input for prediction
    user_input = np.array([[name, online_order, book_table, rate, votes, approx_cost, listed_in_encoded]])

    # Make prediction
    prediction = model.predict(user_input)

    # Return prediction result
    return render_template('index.html', prediction_output=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
