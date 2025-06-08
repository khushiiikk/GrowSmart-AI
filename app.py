from flask import Flask,request,render_template
import numpy as np
import pandas # Though pandas is imported, it's not used in the snippet. Good to have if you plan to use it.
import sklearn # Though sklearn is imported, it's not directly used for functions here.
import pickle

# Load your model and scalers
# Make sure these paths are correct relative to app.py
try:
    model = pickle.load(open('model.pkl','rb'))
    sc = pickle.load(open('standscaler.pkl','rb')) # Corrected variable name from 'sc' to 'stand_scaler' for clarity if you used 'stand_scaler' earlier, but 'sc' is fine if that's what you used.
    mx = pickle.load(open('minmaxscaler.pkl','rb')) # Corrected variable name from 'mx' to 'minmax_scaler' for clarity, but 'mx' is fine.
except FileNotFoundError as e:
    print(f"Error loading model or scalers: {e}. Ensure model.pkl, standscaler.pkl, and minmaxscaler.pkl are in the same directory as app.py")
    # You might want to handle this more robustly in a production environment,
    # e.g., by returning an error message to the user.
    model = None
    sc = None
    mx = None


app = Flask(__name__)

@app.route('/')
def index():
    # When the page is first loaded, 'result' is not available, so the card won't show.
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    # Check if models/scalers are loaded
    if model is None or sc is None or mx is None:
        return render_template('index.html', result="Error: ML models/scalers not loaded. Please check server logs.")

    try:
        # Convert all input values to float immediately
        # Use request.form.get() to safely get values, though direct access is fine if you control the form.
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus']) # Note: Typo 'Phosporus' in your code, keep it consistent with HTML
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

    except ValueError:
        return render_template('index.html', result="Invalid input. Please enter numeric values for all fields.")
    except KeyError as e:
        return render_template('index.html', result=f"Missing form data: {e}. Ensure all input fields in HTML have correct 'name' attributes.")


    # Create a numpy array from the numeric feature list
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Apply scaling
    # Ensure the order of scaling (minmax then standard) matches your training pipeline
    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)


    # Make prediction
    prediction = model.predict(sc_mx_features)

    # Define crop dictionary
    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index.html',result = result)


if __name__ == "__main__":
    app.run(debug=True)