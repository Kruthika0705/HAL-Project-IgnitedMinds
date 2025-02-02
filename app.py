from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import ast
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

app = Flask(__name__)

# Load dataset
file_path = "C:/Users/Joshika K/Downloads/Finalized_dataset.csv"  # Adjust as needed
df = pd.read_csv(file_path)

# Safe conversion of 'Product' column to list
def safe_eval(val):
    try:
        return ast.literal_eval(val) if isinstance(val, str) else []
    except:
        return []

df['Product_List'] = df['Product'].apply(safe_eval)
df['Product_Count'] = df['Product_List'].apply(len)

# Ensure 'Discount_Applied' is integer
df['Discount_Applied'] = df['Discount_Applied'].astype(int)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Season']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Generate synthetic wastage data (20% of samples)
np.random.seed(42)
df['Wastage'] = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])

# Estimate demand dynamically if 'Predicted_Demand' is missing
if 'Predicted_Demand' not in df.columns:
    df['Predicted_Demand'] = df['Total_Items'] * (0.5 + 0.5 * np.random.rand(len(df)))  # Scaled demand estimate

# Add 'Days_Until_Expiration' as a feature
df['Days_Until_Expiration'] = np.random.randint(1, 30, size=len(df))  # Placeholder, ideally from dataset

# Train Random Forest models once at startup
X = df[['Total_Items', 'Product_Count', 'Season', 'Days_Until_Expiration']]
y_wastage = df['Wastage']
y_demand = df['Predicted_Demand']

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X, y_wastage)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X, y_demand)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    product_name = request.form['product_name']
    expiration_date_input = request.form['expiration_date']
    seasonality_input = request.form['seasonality']
    total_items_input = request.form['total_items']

    try:
        total_items = int(total_items_input)
        expiration_date = datetime.strptime(expiration_date_input, "%d-%m-%Y")
        current_date = datetime.now()
        days_until_expiration = max(0, (expiration_date - current_date).days)

        if seasonality_input in label_encoders['Season'].classes_:
            season_encoded = label_encoders['Season'].transform([seasonality_input])[0]
        else:
            season_encoded = -1  # Default or unknown season

        input_data = {
            'Total_Items': total_items,
            'Product_Count': 1,
            'Season': season_encoded,
            'Days_Until_Expiration': days_until_expiration
        }

        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=X.columns, fill_value=0)

        wastage_pred = rf_classifier.predict(input_df)
        waste_prob = rf_classifier.predict_proba(input_df)[:, 1] if rf_classifier.n_classes_ > 1 else np.array([0.0])
        demand_pred = rf_regressor.predict(input_df)[0]

        if waste_prob[0] > 0.05 and total_items > 0:
            recommended_action = 'Discount'
            dynamic_price_adjustment = "Decrease Price"
        else:
            recommended_action = 'Donation'
            dynamic_price_adjustment = "No Price Change"

        base_price_per_item = 100  
        adjusted_price_per_item = base_price_per_item

        if days_until_expiration <= 3:
            adjusted_price_per_item *= 0.7  
        elif days_until_expiration <= 7:
            adjusted_price_per_item *= 0.9  

        if total_items > 10:
            adjusted_price_per_item *= 0.95  
        elif total_items < 3:
            adjusted_price_per_item *= 1.1  

        return render_template('result.html', product_name=product_name,
                               expiration_date=expiration_date.strftime('%d-%m-%Y'),
                               days_until_expiration=days_until_expiration,
                               total_items=total_items,
                               wastage_prediction='Yes' if wastage_pred[0] == 1 else 'No',
                               waste_probability=waste_prob[0],
                               predicted_demand=demand_pred,
                               recommended_action=recommended_action,
                               adjusted_price_per_item=adjusted_price_per_item)

    except ValueError as e:
        return render_template('index.html', error="Invalid input! Please ensure all inputs are correct.")
    
@app.route('/donation')
def donation():
    return render_template('donation.html')

if __name__ == "__main__":
    app.run(debug=True)
