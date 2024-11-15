from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the scaler
scaler = joblib.load('scaler.pkl')
linear_model = joblib.load('linear_regression_model.pkl')
scratch_scaler = joblib.load('scratch_scaler.pkl')
scratch_linear_regression=joblib.load('scratch_linear_regression.pkl')

@app.route("/", methods=["GET", "POST"])
def form():
    if request.method == "POST":
        # Get the input data from the form and convert to appropriate data types
        user_data = {
            'gender': int(request.form.get("gender", 0)),
            'age': int(request.form.get("age", 0)),
            'height(cm)': int(request.form.get("height", 0)),
            'weight(kg)': int(request.form.get("weight", 0)),
            'waist(cm)': float(request.form.get("waist", 0)),
            'eyesight(left)': float(request.form.get("eyesight_left", 0)),
            'eyesight(right)': float(request.form.get("eyesight_right", 0)),
            'hearing(left)': int(request.form.get("hearing_left", 0)),
            'hearing(right)': int(request.form.get("hearing_right", 0)),
            'systolic': int(request.form.get("systolic", 0)),
            'relaxation': int(request.form.get("relaxation", 0)),
            'fasting blood sugar': int(request.form.get("fasting_blood_sugar", 0)),
            'Cholesterol': int(request.form.get("cholesterol", 0)),
            'triglyceride': int(request.form.get("triglyceride", 0)),
            'HDL': int(request.form.get("HDL", 0)),
            'LDL': int(request.form.get("LDL", 0)),
            'hemoglobin': float(request.form.get("hemoglobin", 0)),
            'Urine protein': int(request.form.get("urine_protein", 0)),
            'serum creatinine': float(request.form.get("serum_creatinine", 0)),
            'AST': int(request.form.get("AST", 0)),
            'ALT': int(request.form.get("ALT", 0)),
            'Gtp': int(request.form.get("GTP", 0)),
            'oral': int(request.form.get("oral", 0)),
            'dental caries': int(request.form.get("dental_caries", 0)),
            'tartar': int(request.form.get("tartar", 0))
        }
        # Convert user data into a DataFrame
        user_data_df = pd.DataFrame([user_data])
    
        # Scale the data using the loaded scaler
        user_data_scaled = pd.DataFrame(scaler.transform(user_data_df), columns=user_data_df.columns)
        # Rename columns in the scaled data
        scaled_data = user_data_scaled.to_dict(orient="records")[0]
        
        # Rename the keys to match the expected format in the template
        scaled_data["height"] = scaled_data.pop("height(cm)")  # Rename 'height(cm)' to 'height'
        scaled_data["weight"] = scaled_data.pop("weight(kg)")  # Rename 'weight(kg)' to 'weight'
        scaled_data["waist"] = scaled_data.pop("waist(cm)")  # Rename 'waist(cm)' to 'waist'
        scaled_data["eyesight_left"] = scaled_data.pop("eyesight(left)")  # Rename 'eyesight(left)' to 'eyesight_left'
        scaled_data["eyesight_right"] = scaled_data.pop("eyesight(right)")  # Rename 'eyesight(right)' to 'eyesight_right'
        scaled_data["hearing_left"] = scaled_data.pop("hearing(left)")  # Rename 'hearing(left)' to 'hearing_left'
        scaled_data["hearing_right"] = scaled_data.pop("hearing(right)")  # Rename 'hearing(right)' to 'hearing_right'
        scaled_data["fasting_blood_sugar"] = scaled_data.pop("fasting blood sugar")  # Rename 'fasting blood sugar' to 'fasting_blood_sugar'
        scaled_data["urine_protein"] = scaled_data.pop("Urine protein")  # Rename 'Urine protein' to 'urine_protein'
        scaled_data["serum_creatinine"] = scaled_data.pop("serum creatinine")  # Rename 'serum creatinine' to 'serum_creatinine'
        scaled_data["dental_caries"] = scaled_data.pop("dental caries")  # Rename 'dental caries' to 'dental_caries'
        
        smokingStatus=linear_model.predict(user_data_scaled)
        scaled_data["smoking_status"]=smokingStatus
        print(smokingStatus)
        
        
        #for Scratch Linear Regression
        
    # Add bias term (column of 1s) to the scaled data
        X_bias = np.c_[np.ones(user_data_scaled.shape[0]), user_data_scaled]

        # Predict using the trained model (theta)
        smokingStatus_scratch = X_bias.dot(scratch_linear_regression)  # Linear regression prediction (X * theta)
        scaled_data["smoking_status_scratch"]=smokingStatus_scratch
        return render_template("result.html", **scaled_data)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
