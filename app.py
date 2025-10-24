from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load dataset
data = pd.read_csv('data.csv')
features = ['LotArea', 'OverallQual', 'YearBuilt', 'TotalBsmtSF', 
            'GrLivArea', 'FullBath', 'BedroomAbvGr', 'GarageCars']

# Train model once when app starts
X = data[features].fillna(0)
y = data['SalePrice']
model = LinearRegression()
model.fit(X, y)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values from form
        input_data = [
            float(request.form['LotArea']),
            float(request.form['OverallQual']),
            float(request.form['YearBuilt']),
            float(request.form['TotalBsmtSF']),
            float(request.form['GrLivArea']),
            float(request.form['FullBath']),
            float(request.form['BedroomAbvGr']),
            float(request.form['GarageCars'])
        ]

        # Make prediction
        prediction = model.predict([input_data])[0]

        # Render result page
        return render_template('result.html', prediction=prediction)

    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    app.run(debug=True)
