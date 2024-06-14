import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the trained model, scaler, and label encoder
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Title
st.title('Depression State Prediction')
#Description
st.markdown("""
### Input Descriptions:
- **Sleep**: Frequency of sleep disturbances (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
- **Appetite**: Changes in appetite (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
- **Interest**: Loss of interest in activities (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
- **Fatigue**: Feelings of fatigue or low energy (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
- **Worthlessness**: Feelings of worthlessness or excessive guilt (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
- **Concentration**: Difficulty concentrating (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
- **Agitation**: Physical agitation (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
- **Suicidal Ideation**: Thoughts of self-harm or suicide (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
- **Sleep Disturbance**: Issues with sleeping (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
- **Aggression**: Feelings of aggression (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
- **Panic Attacks**: Experiencing panic attacks (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
- **Hopelessness**: Feelings of hopelessness (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
- **Restlessness**: Feelings of restlessness (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
- **Low Energy**: Lack of energy (1: Never, 2: Always, 3: Often, 4: Rarely, 5: Sometimes, 6: Not at all).
""")


st.sidebar.header('Input Your Parameters')
def user_input_features():
    Sleep = st.sidebar.slider('Sleep', 1, 5, 3)
    Appetite = st.sidebar.slider('Appetite', 1, 5, 3)
    Interest = st.sidebar.slider('Interest', 1, 5, 3)
    Fatigue = st.sidebar.slider('Fatigue', 1, 5, 3)
    Worthlessness = st.sidebar.slider('Worthlessness', 1, 5, 3)
    Concentration = st.sidebar.slider('Concentration', 1, 5, 3)
    Agitation = st.sidebar.slider('Agitation', 1, 5, 3)
    Suicidal_Ideation = st.sidebar.slider('Suicidal Ideation', 1, 5, 3)
    Sleep_Disturbance = st.sidebar.slider('Sleep Disturbance', 1, 5, 3)
    Aggression = st.sidebar.slider('Aggression', 1, 5, 3)
    Panic_Attacks = st.sidebar.slider('Panic Attacks', 1, 5, 3)
    Hopelessness = st.sidebar.slider('Hopelessness', 1, 5, 3)
    Restlessness = st.sidebar.slider('Restlessness', 1, 5, 3)
    Low_Energy = st.sidebar.slider('Low Energy', 1, 5, 3)

    data = {
        'Sleep': Sleep,
        'Appetite': Appetite,
        'Interest': Interest,
        'Fatigue': Fatigue,
        'Worthlessness': Worthlessness,
        'Concentration': Concentration,
        'Agitation': Agitation,
        'Suicidal Ideation': Suicidal_Ideation,
        'Sleep Disturbance': Sleep_Disturbance,
        'Aggression': Aggression,
        'Panic Attacks': Panic_Attacks,
        'Hopelessness': Hopelessness,
        'Restlessness': Restlessness,
        'Low Energy': Low_Energy
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()


input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)

prediction_label = label_encoder.inverse_transform(prediction.astype(int))

st.subheader('Prediction')
st.write(f'The predicted depression state is: {prediction_label[0]}')
