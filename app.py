from flask import Flask, request, render_template
import joblib

# Load the pretrained model
model = joblib.load('iris_knn_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from the request
        feature1 = float(request.form.get('feature1', 0))
        feature2 = float(request.form.get('feature2', 0))
        feature3 = float(request.form.get('feature3', 0))
        feature4 = float(request.form.get('feature4', 0))

        # Make prediction using the loaded model
        prediction = model.predict([[feature1, feature2, feature3, feature4]])

        # Convert numerical prediction to a human-readable label
        if prediction[0] == 0:
            final_pred = 'Setosa'
        elif prediction[0] == 1:
            final_pred = 'Versicolor'
        elif prediction[0] == 2:
            final_pred = 'Virginica'
        else:
            final_pred = None

        return render_template('index2.html', prediction=final_pred)
    
    except Exception as e:
        return render_template('index2.html', prediction=None, error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
