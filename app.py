from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

app = Flask(__name__)
app.static_folder = 'static'
app.template_folder = 'templates'

MODEL_FILE = 'pcos_model_rf.joblib'
ENCODER_FILE = 'label_encoders_rf.joblib'
DATA_FILE_1 = 'PCOS_data_without_infertility.csv'
DATA_FILE_2 = 'PCOS_infertility.csv'

def clean_numeric_value(value):
    if isinstance(value, str):
        cleaned_value = value.rstrip('.')
        try:
            return float(cleaned_value)
        except ValueError:
            return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan

def load_and_preprocess_data():
    try:
        df1 = pd.read_csv(DATA_FILE_1)
        df2 = pd.read_csv(DATA_FILE_2)

        filter_df1 = df1.loc[541:]
        df1.drop(index=filter_df1.index, inplace=True)

        data = pd.merge(df1, df2, on='Patient File No.', suffixes=('', '_y'), how='left')
        data = data.drop(['Unnamed: 44', 'Sl. No_y', 'PCOS (Y/N)_y',
                          '  I   beta-HCG(mIU/mL)_y', 'II   beta-HCG(mIU/mL)_y',
                          'AMH(ng/mL)_y'], axis=1, errors='ignore')

        for col in ["AMH(ng/mL)", "II   beta-HCG(mIU/mL)"]:
            if col in data.columns:
                data[col] = data[col].apply(clean_numeric_value)

        for col in data.columns:
            if data[col].dtype == 'object':
                if not data[col].mode().empty:
                    data[col] = data[col].fillna(data[col].mode()[0])
                else:
                    data[col] = data[col].fillna('')
            else:
                numerical_values = data[col].dropna()
                if not numerical_values.empty:
                    data[col] = data[col].fillna(numerical_values.median())
                else:
                    data[col] = data[col].fillna(0)

        data['Fast food (Y/N)'] = data['Fast food (Y/N)'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
        data['Reg.Exercise(Y/N)'] = data['Reg.Exercise(Y/N)'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

        label_encoders = {}
        for col in data.columns:
            if data[col].dtype == 'object':
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                label_encoders[col] = le

        new_features = ['Pulse rate(bpm)', 'RR (breaths/min)', 'Cycle(R/I)']
        for feature in new_features:
            if feature not in data.columns:
                data[feature] = 0

        X = data.drop(["PCOS (Y/N)", "Sl. No", "Patient File No."], axis=1, errors='ignore')
        y = data["PCOS (Y/N)"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

        rfc = RandomForestClassifier(random_state=43)
        rfc.fit(X_train, y_train)

        joblib.dump(rfc, MODEL_FILE)
        joblib.dump(label_encoders, ENCODER_FILE)
        print("Trained Random Forest model and encoders saved.")
        return joblib.load(MODEL_FILE), joblib.load(ENCODER_FILE), X.columns.tolist()

    except FileNotFoundError as e:
        print(f"Error: Data file not found: {e}")
        return None, None, None
    except Exception as e:
        print(f"An error occurred during data loading and preprocessing: {e}")
        return None, None, None

model, label_encoders, feature_names = load_and_preprocess_data()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST' and model is not None and label_encoders is not None and feature_names is not None:
        try:
            # Retrieving the form inputs and cleaning them
            age = float(request.form.get('age', 0))
            weight = float(request.form.get('weight', 0))
            height_cm = float(request.form.get('height', 0))
            bmi = float(request.form.get('bmi', 0))
            cycle = float(request.form.get('cycle', 0))
            cycle_ri = float(request.form.get('cycle_ri', 0))
           # cycleRegularity = request.form.get('cycleRegularity', '')
            fastFood = 1 if request.form.get('Fast Food') == 'Yes' else 0
            exercise = 1 if request.form.get('Reg.Exercise(Y/N)') == 'Yes' else 0
            waist = float(request.form.get('waist', 0))
            hip = float(request.form.get('hip', 0))
            pulseRate = float(request.form.get('pulseRate', 0))
            rr = float(request.form.get('rr', 0))
            pimples = 1 if 'Pimples' in request.form.getlist('symptoms') else 0
            hairGrowth = 1 if 'Hair Growth' in request.form.getlist('symptoms') else 0
            darkSpots = 1 if 'darkSpots' in request.form.getlist('symptoms') else 0
            hairLoss = 1 if 'Hair Loss' in request.form.getlist('symptoms') else 0
            weightGain = 1 if 'Weight Gain' in request.form.getlist('symptoms') else 0

            input_data = {
                'Age': age,
                'Weight': weight,
                'Height': height_cm,
                'BMI': bmi,
                'Cycle length (days)': cycle,
                'Cycle(R/I)': cycle_ri,
                #'Cycle Regularity': cycleRegularity,
                'Fast food (Y/N)': fastFood,
                'Reg.Exercise(Y/N)': exercise,
                'Waist Circumference (inches)': waist,
                'Hip Circumference (inches)': hip,
                'Pulse rate(bpm)': pulseRate,
                'RR (breaths/min)': rr,
                'Pimples': pimples,
                'Hair Growth': hairGrowth,
                'darkSpots': darkSpots,
                'Hair Loss': hairLoss,
                'Weight Gain': weightGain,
            }

            input_df = pd.DataFrame([input_data])

            # Apply label encoding
            for col, encoder in label_encoders.items():
                if col in input_df.columns:
                    try:
                        input_df[col] = encoder.transform(input_df[col])
                    except ValueError as e:
                        print(f"Error encoding input for column '{col}': {e}")
                        input_df[col] = encoder.transform([input_df[col].iloc[0]])

            # Ensure all features are present
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0

            input_df = input_df[feature_names]

            # Prediction and probability calculation
            prediction_proba = model.predict_proba(input_df)[0][1]
            prediction_threshold = 0.5
            if prediction_proba >= prediction_threshold:
                prediction = f"High Risk (Probability: {prediction_proba:.2f})"
            else:
                prediction = f"Low Risk (Probability: {prediction_proba:.2f})"

        except ValueError as e:
            prediction = f"Invalid input: {e}"
            print(prediction)
        except KeyError as e:
            prediction = f"Missing input field: {e}"
            print(prediction)
        except Exception as e:
            prediction = f"An unexpected error occurred: {e}"
            print(prediction)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)