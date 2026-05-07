import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
import pickle

def main():
    print("Loading datasets...")
    # Load the new single dataset
    data = pd.read_csv('Banglore_traffic_Dataset.csv')

    print("Cleaning data...")
    # Drop Date and Environmental Impact if they exist (to avoid data leakage as analyzed)
    cols_to_drop = ['Date', 'Environmental Impact']
    data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])

    # Our targets
    targets = ['Traffic Volume', 'Average Speed']
    
    # Drop targets from X
    X = data.drop(columns=targets)
    y_volume = data['Traffic Volume']
    y_speed = data['Average Speed']

    # Identify categorical and numerical columns
    categorical_cols = ['Area Name', 'Road/Intersection Name', 'Weather Conditions', 'Roadwork and Construction Activity']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    print("Encoding categorical variables...")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_categorical = encoder.fit_transform(X[categorical_cols])
    
    encoded_col_names = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_categorical, columns=encoded_col_names)
    
    # Combine numerical and encoded categorical features
    X_processed = pd.concat([X[numerical_cols].reset_index(drop=True), encoded_df], axis=1)

    print(f"Training models on {len(X_processed)} rows and {len(X_processed.columns)} features...")
    
    # Model for Traffic Volume
    model_vol = DecisionTreeRegressor(max_depth=10, random_state=42)
    model_vol.fit(X_processed, y_volume)
    
    # Model for Average Speed
    model_spd = DecisionTreeRegressor(max_depth=10, random_state=42)
    model_spd.fit(X_processed, y_speed)
    
    print("Models trained.")

    # Save options for the UI
    dropdown_options = {col: sorted(data[col].astype(str).unique().tolist()) for col in categorical_cols}
    
    # Save min/max for numerical inputs
    numerical_limits = {col: {'min': float(data[col].min()), 'max': float(data[col].max()), 'mean': float(data[col].mean())} for col in numerical_cols}

    # Save to disk
    output_data = {
        'model_vol': model_vol,
        'model_spd': model_spd,
        'encoder': encoder,
        'options': dropdown_options,
        'numerical_limits': numerical_limits,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'features': X_processed.columns.tolist()
    }
    
    print("Saving models to model.pkl...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(output_data, f)
        
    print("Training complete! model.pkl is ready.")

if __name__ == "__main__":
    main()
