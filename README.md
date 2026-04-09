# 🚗 Bengaluru Traffic Predictor

An interactive web application that predicts traffic travel times in Bengaluru using Machine Learning. Built with Python, `scikit-learn` (Random Forest), and a dynamic front-end using `Streamlit`. 

## 🌟 Features

* **🕰️ Time Travel Mode:** Simulate future trips by explicitly selecting a Day of the Week, Time of Day, and forecasted Weather to see how long your commute will take between two locations.
* **🤔 What-If Analysis:** Want to see the impact of rain on your current route? Fix a base route and toggle specific atmospheric or timing constraints (e.g., Clear vs. Heavy Rain, Weekday vs. Weekend) to instantly compare the time differential.
* **⚖️ Comparison Mode:** Perform robust A/B testing on entirely different situational layouts with side-by-side interactive metrics and bar comparisons.

## 💻 Tech Stack
- **Languages:** Python
- **Machine Learning:** `scikit-learn` (RandomForestRegressor), `pandas`
- **Frontend / UI:** Streamlit (with injected HTML/CSS for modern dark-mode aesthetics)
- **Serialization:** `pickle`

## 📁 Project Structure

```text
├── app.py                            # Main Streamlit frontend application
├── train_model.py                    # Script to merge data, preprocess, and train the ML model
├── bengaluru_traffic_features.csv    # Static dataset: Traffic Features
├── bengaluru_traffic_target.csv      # Static dataset: Target Variables (Travel time)
├── model.pkl                         # Saved ML model and Encoders (Generated after training)
└── README.md                         # Project documentation
```

## 🚀 Setup & Installation

Follow these instructions to get a copy of the project up and running on your local machine.

### 1. Prerequisites
Make sure you have Python 3.8+ installed on your system. 

Install the required Python modules using `pip`:
```bash
python -m pip install pandas scikit-learn streamlit
```

### 2. Train the Model
The application relies on a pre-trained `.pkl` file to function rapidly on the frontend. To generate exactly that, just run the training script once:
```bash
python train_model.py
```
*Note: Ensure both CSV datasets are in the same folder before running this script. The script handles data-merging, cleaning, and model output automatically.*

### 3. Run the Web Application
Once `model.pkl` is successfully generated, you can launch the Streamlit server:
```bash
python -m streamlit run app.py
```

Streamlit will automatically open your default browser pointing to `http://localhost:8501`.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome. Feel free to check the issues page if you want to contribute.
