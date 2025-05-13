import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import KNNImputer
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, LayerNormalization, MultiHeadAttention, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tabulate import tabulate

# Load dataset
data = pd.read_csv("/content/drive/MyDrive/all_drives_no_zeros_unique_hr.csv")

# Feature Engineering
data['HR_rolling_mean'] = data['HR_mean'].rolling(window=5, min_periods=1).mean()
data['RESP_rolling_mean'] = data['RESP_mean'].rolling(window=5, min_periods=1).mean()
data['HR_x_RESP'] = data['HR_mean'] * data['RESP_mean']
data['HR_squared'] = data['HR_mean'] ** 2
data.dropna(inplace=True)

features = ['footGSR_mean', 'handGSR_mean', 'HR_mean', 'RESP_mean',
            'HR_rolling_mean', 'RESP_rolling_mean', 'HR_x_RESP', 'HR_squared']
target = 'Stress_mean'

X = data[features]
y = data[target]

# Handle missing values
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
y_imputed = np.nan_to_num(y)

# Feature Selection using SHAP
xgb_temp = XGBRegressor().fit(X_imputed, y_imputed)
explainer = shap.TreeExplainer(xgb_temp)
shap_values = explainer.shap_values(X_imputed)
feature_importance = np.abs(shap_values).mean(axis=0)
selected_features = [features[i] for i in np.argsort(feature_importance)[-8:]]
X = pd.DataFrame(X_imputed, columns=features)[selected_features]

# Data Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
target_transformer = PowerTransformer()
y_transformed = target_transformer.fit_transform(y_imputed.reshape(-1, 1)).flatten()

# Reshape for LSTM & Transformer
X_seq = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_transformed, test_size=0.2, random_state=42)
X_train_seq, X_test_seq, _, _ = train_test_split(X_seq, y_transformed, test_size=0.2, random_state=42)

# Model Building
xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6, random_state=42)

lstm_model = Sequential([
    LSTM(128, activation='relu', input_shape=(X_train_seq.shape[1], 1), return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64, activation='relu', return_sequences=False),
    Dropout(0.3),
    Dense(1)
])
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.2):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_transformer(input_shape, embed_dim=64, num_heads=4, ff_dim=64):
    inputs = Input(shape=input_shape)
    x = Dense(embed_dim)(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

transformer_model = build_transformer(input_shape=(X_train_seq.shape[1], 1))

# Training Models
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

print("Training XGBoost...")
xgb_model.fit(X_train, y_train)

print("Training LSTM...")
lstm_model.fit(X_train_seq, y_train, epochs=50, batch_size=32, validation_data=(X_test_seq, y_test), callbacks=[lr_scheduler])

print("Training Transformer...")
transformer_model.fit(X_train_seq, y_train, epochs=50, batch_size=32, validation_data=(X_test_seq, y_test), callbacks=[lr_scheduler])

# Calculate and display model performance metrics
def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return r2, mse, rmse

# Get predictions
y_pred_xgb = xgb_model.predict(X_test)
y_pred_lstm = lstm_model.predict(X_test_seq).flatten()
y_pred_transformer = transformer_model.predict(X_test_seq).flatten()
y_pred_ensemble = (y_pred_xgb + y_pred_lstm + y_pred_transformer) / 3

# Calculate metrics
metrics_xgb = calculate_metrics(y_test, y_pred_xgb)
metrics_lstm = calculate_metrics(y_test, y_pred_lstm)
metrics_transformer = calculate_metrics(y_test, y_pred_transformer)
metrics_ensemble = calculate_metrics(y_test, y_pred_ensemble)

# Create performance table
performance_table = [
    ["XGBoost", *metrics_xgb],
    ["LSTM", *metrics_lstm],
    ["Transformer", *metrics_transformer],
    ["Ensemble", *metrics_ensemble]
]

# Print performance table
print("\nModel Performance Metrics:")
print(tabulate(performance_table, 
               headers=["Model", "R² Score", "MSE", "RMSE"], 
               tablefmt="grid",
               floatfmt=(".4f", ".4f", ".4f", ".4f")))

# Function for User Input Prediction
def predict_stress_from_series(user_series):
    user_series['HR_rolling_mean'] = user_series['HR_mean'].rolling(window=5, min_periods=1).mean()
    user_series['RESP_rolling_mean'] = user_series['RESP_mean'].rolling(window=5, min_periods=1).mean()
    user_series['HR_x_RESP'] = user_series['HR_mean'] * user_series['RESP_mean']
    user_series['HR_squared'] = user_series['HR_mean'] ** 2
    user_averaged = user_series.mean(axis=0)
    user_input = [user_averaged.get(feature, 0) for feature in selected_features]
    user_input_scaled = scaler.transform(imputer.transform([user_input]))
    user_input_scaled_seq = user_input_scaled.reshape((1, len(selected_features), 1))
    pred_xgb = xgb_model.predict(user_input_scaled)
    pred_lstm = lstm_model.predict(user_input_scaled_seq).flatten()
    pred_transformer = transformer_model.predict(user_input_scaled_seq).flatten()
    pred_ensemble = (pred_xgb + pred_lstm + pred_transformer) / 3
    return target_transformer.inverse_transform([[pred_ensemble[0]]])[0][0]

# Continuous Input Loop
while True:
    user_input = input("\nEnter values for footGSR_mean, handGSR_mean, HR_mean, RESP_mean (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Exiting...")
        break
    try:
        user_values = list(map(float, user_input.split()))
        if len(user_values) != 4:
            raise ValueError("Please enter exactly four numerical values.")

        # Create DataFrame from user input
        user_series = pd.DataFrame([user_values], columns=['footGSR_mean', 'handGSR_mean', 'HR_mean', 'RESP_mean'])

        # Predict stress level
        predicted_stress = predict_stress_from_series(user_series)

        # Prepare table data
        table_data = [
            ["footGSR_mean", user_values[0]],
            ["handGSR_mean", user_values[1]],
            ["HR_mean", user_values[2]],
            ["RESP_mean", user_values[3]],
            ["Predicted Stress Level", f"{predicted_stress:.4f}"]
        ]

        # Print table
        print("\nPrediction Results:")
        print(tabulate(table_data, headers=["Feature", "Value"], tablefmt="grid"))

    except ValueError as e:
        print(f"Error: {e}")