from analysisfunctions import *
from preprocessing import preprocess_data
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoModelForCausalLM
from joblib import load
import streamlit as st
from typing import Optional
import re



clf_freq = load('decision_tree_freq.joblib')
vectorizer_freq = load('tfidf_vectorizer_freq.joblib')
label_encoder_func_freq = load('label_encoder_func_freq.joblib')
clf_con = load('decision_tree_verb.joblib')
vectorizer_con = load('tfidf_vectorizer_verb.joblib')
label_encoder_func_con = load('label_encoder_func_verb.joblib')
clf_trend = load('decision_tree_trend.joblib')
vectorizer_trend = load('tfidf_vectorizer_trend.joblib')
label_encoder_func_trend = load('label_encoder_func_trend.joblib')
        
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
    
    st.title("Developmental Disorder Chatbot")
    st.write("Hallo! Typ 'exit' om te stoppen.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", offload_folder="model")

    user_input = st.sidebar.selectbox(
    "Welk soort vraag is het?", 
    ["frequentie", "verbanden", "trend"])
    
    slider_1 = st.sidebar.slider(
        "Slider 1 - Stel een waarde in voor een cut-off waarde voor 1 enkele stoornis.", 
        min_value=0, max_value=20, value=5, step=1)
    slider_2 = st.sidebar.slider(
        "Slider 2 - Stel een waarde in voor de cut-off waarde overheen meerdere stoornissen.", 
        min_value=0, max_value=20, value=3, step=1)
    slider_3 = st.sidebar.slider(
        "Slider 3 - Stel een waarde in voor het aantal alarmsignalen.", 
        min_value=0, max_value=20, value=1, step=1)

    if user_input.lower() == "exit":
        st.write("Tot ziens!")
        return
    
    if data_file:
        if signal_file:
            data, signal_df = preprocess_data(data_file, signal_file, domain_file, slider_1, slider_2, slider_3)
    else:
        data, signal_df = None, None
    
    specific_question = st.text_input("Dankuwel, wat is de vraag?")
    
    if specific_question:
        if user_input == "frequentie":
            clf = clf_freq
            vectorizer = vectorizer_freq
            label_encoder_func = label_encoder_func_freq
        elif user_input == "verbanden":
            clf = clf_con
            vectorizer = vectorizer_con
            label_encoder_func = label_encoder_func_con
        elif user_input == "trend":
            clf = clf_trend
            vectorizer = vectorizer_trend
            label_encoder_func = label_encoder_func_trend

        X_keywords = vectorizer.transform([specific_question])
        X = X_keywords

        y_pred = clf.predict(X)
        analysis_function_name = label_encoder_func.inverse_transform(y_pred)[0]
        st.write(f"The predicted function name is: {analysis_function_name}")

    
        if analysis_function_name:
            analysis_function = globals().get(analysis_function_name)
            if analysis_function and data is not None:
                analysis_response = analysis_function(data, specific_question, signal_df)
            if isinstance(analysis_response, pd.DataFrame):
                st.table(analysis_response)  
            else:
                st.write(f"Bot: {analysis_response}")
        else:
            st.write("Sorry, ik kan de gevraagde analyse niet vinden.")
    
if __name__ == "__main__":
    run_main_streamlit(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        data_file="data_file.csv"
    )
