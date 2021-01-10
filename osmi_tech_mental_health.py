# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:13:28 2020

@author: Dilini
"""
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def run():
    st.sidebar.title("Machine Learning Model Selector")
    add_model_selectbox = st.sidebar.selectbox(
        "Pick a tuned prediction model",
        ("Catboost", "Random Forest", "Ridge Regression"))

    if add_model_selectbox == "Catboost":
        model = load_model('Tuned_Catboost_07Jan21')
    elif add_model_selectbox == "Random Forest":
        model = load_model('Tuned_RandomForest_07Jan21')
    elif add_model_selectbox == "Ridge Regression":
        model = load_model('Tuned_Ridge_07Jan21')

    def predict(model, input_df):
        predictions_df = predict_model(estimator=model, data=input_df)
        predictions = predictions_df['Label'][0]
        return predictions


    st.sidebar.success('This app was developed using the OSMI mental health in tech data: https://www.kaggle.com/noriuk/us-education-datasets-unification-project') # CHANGE LATER
    
    st.title("OSMI Mental Health")
    st.markdown("Select a model from the sidebar dropdown menu to generate a prediction for an individual's likelihood of bringing up a mental health issue at work.")

    st.info('Move the sliders and click Predict')

    Question_1 = st.select_slider('Are you self-employed? No = 0; Yes = 1', options=[0,1])
    Question_2 = st.select_slider('Do you have previous employers? No = 0; Yes = 1', options=[0,1])
    Question_3 = st.selectbox('Would you be willing to bring up a physical health issue with a potential employer in an interview?',("Maybe","Yes","No"))
    Question_5 = st.selectbox('Do you feel that being identified as a person with a mental health issue would hurt your career?', ("Maybe", "No, I don't think it would", "Yes, I think it would", "No, it has not", "Yes, it has"))
    Question_6 = st.selectbox("Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?", ("No, I don't think they would", "Maybe", "Yes, they do", "Yes, I think they would", "No, they do not"))
    Question_7 = st.selectbox("How willing would you be to share with friends and family that you have a mental illness?", ("Somewhat open", "Neutral", "Not applicable to me (I do not have a mental illness)", "Very open", "Not open at all", "Somewhat not open"))
    Question_8 = st.selectbox("Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?", ("No", "Maybe/Not sure", "Yes, I experienced", "Yes, I observed"))
    Question_9 = st.selectbox("Do you have a family history of mental illness?", ("No", "Yes", "I don't know"))
    Question_10 = st.selectbox("Have you had a mental health disorder in the past?", ("Yes", "Maybe", "No"))
    Question_11 = st.selectbox("Do you currently have a mental health disorder?", ("No","Yes","Maybe"))
    Question_12 = st.selectbox("Have you been diagnosed with a mental health condition by a medical professional?", ("No","Yes"))
    Question_13 = st.select_slider("Have you ever sought treatment for a mental health issue from a mental health professional? No = 0; Yes = 1", options=[0,1])
    Question_14 = st.selectbox("If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?",("Not applicable to me","Rarely","Sometimes","Never","Often"))
    Question_15 = st.selectbox("If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?",("Not applicable to me", "Sometimes", "Often", "Rarely", "Never"))
    Question_16 = st.number_input('What is your age?', min_value=18, max_value=99)
    Question_17 = st.selectbox("What is your gender?", ("MALE", "FEMALE", "Other"))
    Question_18 = st.text_input("What country do you live in?")
    Question_19 = st.text_input("What country do you work in?")
    # Question 20 omitted as it is a multi select question about job types
    Question_21 = st.selectbox("Do you work remotely?", ("Sometimes", "Never", "Always"))

    output=""

    input_dict = {'Question_1' : Question_1,
                  'Question_2' : Question_2,
                  'Question_3' : Question_3,
                  'Question_5' : Question_5,
                  'Question_6' : Question_6,
                  'Question_7' : Question_7,
                  'Question_8' : Question_8,
                  'Question_9' : Question_9,
                  'Question_10' : Question_10,
                  'Question_11' : Question_11,
                  'Question_12' : Question_12,
                  'Question_13' : Question_13,
                  'Question_14' : Question_14,
                  'Question_15' : Question_15,
                  'Question_16' : Question_16,
                  'Question_17' : Question_17,
                  'Question_18' : Question_18,
                  'Question_19' : Question_19,
                  'Question_21' : Question_21}
    input_df = pd.DataFrame([input_dict])


    if st.button("Predict"):
        output = predict(model=model, input_df=input_df)
        output = str(output)

    st.success('Prediction for bringing up mental health at workplace {}'.format(output))

if __name__ == '__main__':
    run()

