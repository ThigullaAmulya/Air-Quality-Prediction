from flask import Flask, request, render_template
import pickle
import numpy as np
 
# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)
 
app = Flask(__name__)
 
# Route for the combined welcome and input form page
@app.route('/')
def home():
    return render_template('index.html')
 
# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    output = 'Poor' if prediction[0] == 1 else 'Good'
 
    # Render the result page with the prediction
    return render_template('result.html', prediction_text=f'Air Quality is: {output}')
 
if __name__ == "__main__":
 app.run(debug=True)
