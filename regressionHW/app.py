import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

# Set page configuration
st.set_page_config(
    page_title="BMI Prediction - Streamlit App",
    page_icon="❤️",
)

# Function to load the pre-trained model
def load_prediction_model(model_file):
    loaded_model = joblib.load(open(model_file, "rb"))
    return loaded_model

# Main function to run the Streamlit app
def main():
    """Simple Linear Regression for BMI Prediction"""
    st.title('BMI Prediction App')

    html_templ = """<div style="background-color:#0047AB;padding:10px;">
                    <h3 style="color:white">Predict BMI Level using Height and Weight</h3>
                    </div>"""
    
    st.markdown(html_templ, unsafe_allow_html=True)
    
    # Sidebar selection
    activity = ["Predict BMI", "What is Linear Regression?"]
    choice = st.sidebar.selectbox("Select Activity", activity)

    if choice == 'Predict BMI':
        st.markdown('---')
        st.subheader('Enter your Height and Weight to predict BMI')
        
        # Input fields for user to enter height and weight
        height = st.number_input('Enter Height (in cm)', min_value=100, max_value=250, step=1)
        weight = st.number_input('Enter Weight (in kg)', min_value=30, max_value=200, step=1)

        # Load pre-trained model
        regressor = load_prediction_model('model/obesity_train.pkl')  # Adjust path if necessary
        
        if st.button('Predict'):
            # Prepare input data for prediction
            input_data = np.array([[height, weight]])

            
            # Make the prediction
            predicted_bmi = regressor.predict(input_data)
            
            # # Map predicted levels to labels
            # mapping = {1: 'Underweight', 2: 'Normal Weight', 3: 'Overweight', 4: 'Obese'}
            # predicted_label = mapping.get(int(predicted_obesity_level[0]), 'Unknown')

            st.info(f'Besar BMI: {predicted_bmi}')   
    else:
        st.markdown('---')
        st.subheader('What is Linear Regression?')
        st.write('')
        st.info('Linear regression is a statistical method for modeling the relationship between a dependent variable '
                '(target) and one or more independent variables (predictors) using a linear equation to make predictions.')

if __name__ == '__main__':
    main()
