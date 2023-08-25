
from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the pre-trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

scaler = StandardScaler()
@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        location = request.form['Location']
        year = int(request.form['year'])
        rural_pop = float(request.form['rural_pop'])
        urban_pop = float(request.form['urban_pop'])
        electric_rural = float(request.form['electric_rural'])

        # Create a DataFrame from the input
        input_data = pd.DataFrame({
            'Location': [location],
            'Year': [year],
            'RuralPop': [rural_pop],
            'UrbanPop': [urban_pop],
            'ElectricRural': [electric_rural]
        })

        # Scale the input data
        input_scaled = scaler.fit_transform(input_data)

        # Predict the value class
        predicted_class = model.predict(input_scaled)
        # predicted_prob = model.predict_proba(input_scaled)

        # Map the predicted class to labels
        predicted_map = {0: 'Low', 1: 'Medium', 2: 'High'}
        predicted_label = predicted_map[predicted_class[0]]


        return render_template('result.html', predictedclass=predicted_label)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)