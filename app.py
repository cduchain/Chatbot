from analysisfunctions import *
from preprocessing import preprocess_data, determine_question_type, filter_question, clean_result
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoModelForCausalLM
from joblib import load
import streamlit as st
from typing import Optional
import re
import time
from itertools import combinations
import logging
import os
import scipy.stats as stats
import numpy as np


clf_both = load('decision_tree_both.joblib')
vectorizer_both = load('tfidf_vectorizer_both.joblib')
label_encoder_func_both = load('label_encoder_func_both.joblib')
clf_comp = load('decision_tree_comp.joblib')
vectorizer_comp = load('tfidf_vectorizer_comp.joblib')
encoder_comp = load('label_encoder_func_comp.joblib')
try:
    hf_token = st.secrets["huggingface"]["token"]
except KeyError:
    st.error("Hugging Face token not found. Please check secrets.toml.")
    st.stop()

def run_main_streamlit(
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",  
    data_file: str = "data_file.csv",
    signal_file: str = "signals.csv",
    domain_file: str = "domain.csv",
    temperature: float = 0.3,
    top_p: float = 0.8,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = 256,
    ):
    
    start_time_load = time.perf_counter()
    st.title("Wegwijzer Ontwikkelingszorgen Data-analyse Chatbot")
    
    function_descriptions = {
    'analyze_freq': 'Hoeveel keer werd er voor elke type van 1 element gekozen?',
    'range_signal' : 'Wat is het leeftijdsinterval voor een specifiek signaal/alle signalen?',
    'most_element': 'Welke type van een element werd het meest gekozen of hoeveel keer werd een specifiek gekozen type van het element gekozen.? Wat zijn de top meest gekozen (alarm)signalen?',
    'least_element': 'Wat is het minste gekozen type van het element?',
    'signals_percentage': 'Welke (alarm)signalen komen voor in minder dan X% van de gevallen? + kan filteren op types van elementen',
    'signal_in_range' : 'Welke (alarm)signalen worden vaker/minder vaak gekozen bij kinderen in de eerste x maanden van het leeftijdsinterval?',
    'combo_function': 'Wat zijn de meest/minst aangeduide signalen/elementen voor de leeftijd x-x maanden, relatie x, ontwikkelingsprobleem x en subdomein x in domein x? + Uit welke (sub)domeinen komen de meeste vragen?',
    'combo_signal' : 'Welke signalen komeen meer/minder dan x% van de tijd smaen voor? Welke signalen komen minst/vaakst samen voor?',
    'how_many_alarm': 'Hoeveel kinderen vertonen ten minste X alarmsignaal?',
    'combo_howmany': 'Hoeveel signalen vertonen kinderen gemiddeld, afhankelijk van gekozen elementen? of Voor hoeveel kinderen gelden die specifieke elementen?',
    'combo_correlations' : 'Wat is de correlatie tussen de specifiek type van element en leeftijd?',
    'time_more': 'Welke signalen zijn in de afgelopen X maanden vaker/minder gemeld dan voorheen? + filteren op types van elementen',
    'time_evolutie_element' : 'Zijn er elementen die vaker gekozen worden sinds X maanden geleden?',
    'time_element' : 'Hoeveel keer werd elk element/specifiek type van element gekozen in laatste x maanden?',
    'time_data': 'Hoeveel nieuwe data is er (elke maand) toegevoegd in de afgelopen X maanden?',
    'combo_atleast': 'Hoeveel kinderen hebben tenminste x (alarm)signalen? + filteren op types van elementen',
    'combo_comparison' : 'Vergelijkt leeftijd of relaties op basis van andere of geen elementen.'}
    
    @st.cache_resource
    def load_model(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", offload_folder="model", token=hf_token)
        return tokenizer, model
    tokenizer, model = load_model(model_name)
    
    slider_1 = st.sidebar.slider(
        "Slider 1 - Stel een waarde in voor een cut-off waarde voor 1 enkele stoornis.", 
        min_value=0, max_value=20, value=5, step=1)
    slider_2 = st.sidebar.slider(
        "Slider 2 - Stel een waarde in voor de cut-off waarde overheen meerdere stoornissen.", 
        min_value=0, max_value=20, value=3, step=1)
    slider_3 = st.sidebar.slider(
        "Slider 3 - Stel een waarde in voor het aantal alarmsignalen.", 
        min_value=0, max_value=20, value=1, step=1)

    @st.cache_data
    def preprocess_data_cached(data_file, signal_file, domain_file, slider_1, slider_2, slider_3):
        return preprocess_data(data_file, signal_file, domain_file, slider_1, slider_2, slider_3)
    data, signal_df = preprocess_data_cached(data_file, signal_file, domain_file, slider_1, slider_2, slider_3)
    
    end_time_load = time.perf_counter()
    specific_question = st.text_input("Hallo! Waarmee kan ik jou helpen?")
    start_time_2 = time.perf_counter()
    
    
    if specific_question:
        specific_question = specific_question.lower()
        specific_question = filter_question(specific_question)
        category = determine_question_type(specific_question)
        st.write(f'herschreven vraag: {specific_question}')
        if category == "other":
            clf = clf_both
            vectorizer = vectorizer_both
            label_encoder_func = label_encoder_func_both
        
        filtered_data = filter_data(data, specific_question)
        if category == "verbanden":   
            if 'per leeftijd' in specific_question:
                analysis_function_name = 'signal_range'
            elif any(word in specific_question for word in ['hoeveel', 'aantal']):
                analysis_function_name = 'combo_howmany'
            elif any(word in specific_question for word in ['ten minste', 'minstens', 'minimum']):
                analysis_function_name = 'combo_atleast'
            elif any(word in specific_question for word in ['correlatie', 'correlaties']):
                analysis_function_name = 'combo_correlation'
            elif any(word in specific_question for word in ['%', 'procent']):
                analysis_function_name = 'signals_percentage'
            elif any(word in specific_question for word in ['afgelopen', 'laatste', 'voorbije', 'vorige']):
                analysis_function_name = 'time_more'
            elif any(word in specific_question for word in ['tegelijk', 'samen', 'hetzelfde moment','alleen', 'uitgezonderd', 'individueel']):
                analysis_function_name = 'combo_signal'
            elif ('meer' in specific_question and 'dan' in specific_question) or ('minder' in specific_question and 'dan' in specific_question) or ('vergelijk' in specific_question):
                analysis_function_name = 'combo_comparison'
            else:
                analysis_function_name = 'combo_function'
            

        if category == 'other':
            confidence_threshold = 0.7
            X_keywords = vectorizer.transform([specific_question])
            X = X_keywords

            y_prob = clf.predict_proba(X)  
            max_prob = np.max(y_prob) 
            y_pred = clf.predict(X)
            analysis_function_name = label_encoder_func.inverse_transform(y_pred)[0]
            feature_names = vectorizer.get_feature_names_out()
            important_words = [feature_names[i] for i in X.nonzero()[1]]
            
            st.write(f"Voorspelde functie: {analysis_function_name}\n")
            st.write(f"Waarom deze functie? De chatbot vond de volgende belangrijke woorden in jouw vraag: {', '.join(important_words)}\n")

            if max_prob < confidence_threshold:
                st.write("**ik ben niet zeker welke functie bij je vraag past**")

        end_time_2 = time.perf_counter()
        elapsed_time_2 = end_time_2 - start_time_2
        start_time = time.perf_counter()

        if analysis_function_name == 'combo_comparison':
            analysis_function = globals().get(analysis_function_name)
            analysis_response = analysis_function(data, specific_question, signal_df, clf_comp, vectorizer_comp, encoder_comp)
            function_description = function_descriptions.get(analysis_function_name, "No description available.")
            if isinstance(analysis_response, pd.DataFrame):
                st.table(analysis_response)  
            else:
                analysis_response = str(analysis_response).strip()
                st.write(f"**{analysis_response}**") 
        else:
            if analysis_function_name:
                function_description = function_descriptions.get(analysis_function_name, "No description available.")
                analysis_function = globals().get(analysis_function_name)
                if analysis_function and data is not None:
                    analysis_response = analysis_function(data, specific_question, signal_df)
                    analysis_response = clean_result(analysis_response)
                    if isinstance(analysis_response, pd.DataFrame):
                        st.table(analysis_response)  
                    else:
                        analysis_response = str(analysis_response).strip()
                        st.write(f"**{analysis_response}**")
                elapsed_time_3 = end_time_load - start_time_load 
                end_time = time.perf_counter()  
                elapsed_time = end_time - start_time 

            st.write(f"Voorspelde functie: {analysis_function_name}\n")
            st.write(f"Wat doet deze functie? {function_description}")
            st.write(f"De analyse duurde {elapsed_time:.2f} seconden.")
            st.write(f"De functie vinden duurde {elapsed_time_2:.2f} seconden.")
            st.write(f"De chatbot laden duurde {elapsed_time_3:.2f} seconden.")
        
if __name__ == "__main__":
    run_main_streamlit(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        data_file="data_file.csv")
