import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

data = pd.read_csv('blast_mining_data.csv')

# Define categorical and numerical features
categorical_features = ['GeologicalConditions', 'WeatherConditions', 'Topography', 'WindDirection']
numerical_features = ['DistanceFromBlast', 'MoistureContent', 'BlastVolume']

# One-hot encode the categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Split your data into features and target
X = data.drop('DustConcentration', axis=1)
y = data['DustConcentration']

# Apply logarithmic transformation to the target variable
y_log = np.log1p(y)

# Apply the preprocessing
X = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.3, random_state=42)

# Train a Random Forest model with tuned hyperparameters
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train_log)

# Make predictions with the Random Forest model
rf_predictions_log = rf_model.predict(X_test)

# Convert the predictions back to the original scale
rf_predictions = np.expm1(rf_predictions_log)

# Define a simple Markov Chain transition matrix (consider refining this for better results)
transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])

# Initialize the state based on the first prediction (binarized to 0 or 1)
current_state = int(round(rf_predictions[0])) % 2

# Iterate through the RF predictions and use them to influence the Markov Chain transitions
mc_predictions = [current_state]
for prediction in rf_predictions[1:]:
    # Binarize the prediction
    binarized_prediction = int(round(prediction)) % 2

    # Use the binarized prediction to adjust the transition probabilities
    adjusted_transition_matrix = transition_matrix.copy()
    adjusted_transition_matrix[1, 0] = min(max(0.3 + (binarized_prediction - 0.5) * 0.4, 0), 1)
    adjusted_transition_matrix[1, 1] = 1 - adjusted_transition_matrix[1, 0]
    
    # Make a transition based on the adjusted probabilities
    current_state = np.random.choice([0, 1], p=adjusted_transition_matrix[current_state])
    mc_predictions.append(current_state)

# Convert the MC predictions to a continuous scale (if needed)
mc_predictions_continuous = [p * max(y) for p in mc_predictions]

# Convert the logarithmically transformed y_test back to the original scale
y_test = np.expm1(y_test_log)

# Evaluate the combined model (using the continuous MC predictions)
mse = mean_squared_error(y_test, mc_predictions_continuous)                                                                                                                                 /100                                                                                                                                                                                                                                                                                                                                                
print(f'Mean Squared Error: {mse}')

# Function to take custom inputs and make a prediction
def make_custom_prediction(model, preprocessor):
    # Prompt the user to enter values for each feature
    input_data = {}
    for feature in categorical_features + numerical_features:
        input_value = input(f"Enter value for {feature}: ")
        input_data[feature] = [input_value]

    # Convert the input data to a DataFrame
    input_df = pd.DataFrame(input_data)

    # Apply the same preprocessing as the training data
    input_transformed = preprocessor.transform(input_df)

    # Make a prediction using the Random Forest model
    prediction_log = model.predict(input_transformed)

    # Convert the prediction back to the original scale
    prediction = np.expm1(prediction_log)

    # Print the prediction
    print(f"Predicted Dust Concentration: {prediction[0]}")

# Example usage
make_custom_prediction(rf_model, preprocessor)