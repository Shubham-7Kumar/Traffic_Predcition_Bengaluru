import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import pickle
import os

def main():
    print("Loading datasets...")
    features_df = pd.read_csv('bengaluru_traffic_features.csv')
    target_df = pd.read_csv('bengaluru_traffic_target.csv')

    print("Merging datasets...")
    # Merge on Trip_ID
    data = pd.merge(features_df, target_df, on="Trip_ID")

    print("Preparing distance mapping...")
    # Map average distance for start-end pairs so the app can auto-fill this
    distance_mapping = data.groupby(['start_area', 'end_area'])['distance_km'].mean().to_dict()

    # Drop columns not used for training
    # 'average_speed_kmph' is essentially leakage if we are predicting travel time
    columns_to_drop = ['Trip_ID', 'average_speed_kmph', 'travel_time_minutes']
    
    X = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    y = data['travel_time_minutes']

    # Define categorical columns
    categorical_cols = ['start_area', 'end_area', 'time_of_day', 'day_of_week', 
                        'weather_condition', 'traffic_density_level', 'road_type']
    
    numerical_cols = ['distance_km']

    print("Encoding categorical variables...")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_categorical = encoder.fit_transform(X[categorical_cols])
    
    # Get feature names for interpretability in the dataframe
    encoded_col_names = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_categorical, columns=encoded_col_names)
    
    # Combine numerical and encoded categorical features
    X_processed = pd.concat([X[numerical_cols].reset_index(drop=True), encoded_df], axis=1)

    print(f"Training Random Forest model on {len(X_processed)} rows and {len(X_processed.columns)} features...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_processed, y)
    print("Model trained.")

    # Unique values for dropdowns in frontend
    dropdown_options = {
        'start_areas': sorted(data['start_area'].unique().tolist()),
        'end_areas': sorted(data['end_area'].unique().tolist()),
        'time_of_day': sorted(data['time_of_day'].unique().tolist()),
        'day_of_week': sorted(data['day_of_week'].unique().tolist()),
        'weather_condition': sorted(data['weather_condition'].unique().tolist()),
        'traffic_density_level': sorted(data['traffic_density_level'].unique().tolist()),
        'road_types': sorted(data['road_type'].unique().tolist())
    }

    # Save to disk
    output_data = {
        'model': model,
        'encoder': encoder,
        'distance_mapping': distance_mapping,
        'options': dropdown_options,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'features': X_processed.columns.tolist()
    }
    
    print("Saving model to model.pkl...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(output_data, f)
        
    print("Training complete! model.pkl is ready.")

if __name__ == "__main__":
    main()
