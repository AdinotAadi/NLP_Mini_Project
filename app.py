import streamlit as st
from PIL import Image
import easyocr
import numpy as np
from transformers import pipeline

# Load the sentiment analysis and NER pipelines
sentiment_pipe = pipeline("text-classification", model="LondonStory/txlm-roberta-hindi-sentiment")
ner_pipe = pipeline("token-classification", model="ai4bharat/IndicNER")

# Function to extract text from images
def extract_text_from_image(image_file):
    reader = easyocr.Reader(['hi'])  # Add languages as needed
    extracted_text = ""
    try:
        image = Image.open(image_file)
        image_np = np.array(image)
    except IOError:
        return extracted_text
    text_list = reader.readtext(image_np, detail=0)
    for i in text_list:
        extracted_text += " " + str(i)
    return extracted_text.strip()

# Function to process the file and extract text
def process_file(file):
    extracted_text = ""
    if file.type == "text/plain":
        # If it's a text file, read it directly
        extracted_text = file.read().decode("utf-8")
    elif file.type in ["image/png", "image/jpeg"]:
        # If it's an image, extract text using OCR
        extracted_text = extract_text_from_image(file)
    else:
        st.error("Unsupported file type")
    return extracted_text

# Function to display sentiment in color-coded boxes
def display_sentiment(sentiment_result):
    for sentiment in sentiment_result:
        label = sentiment['label']
        score = sentiment['score']
        
        # Choose box color based on the label
        if label == "LABEL_0":  # Negative sentiment
            color = "red"
            label_text = "Negative Sentiment"
        elif label == "LABEL_1":  # Neutral sentiment
            color = "blue"
            label_text = "Neutral Sentiment"
        elif label == "LABEL_2":  # Positive sentiment
            color = "green"
            label_text = "Positive Sentiment"
        
        # Display the sentiment in a colored box
        st.markdown(
            f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; color: white;'>"
            f"<strong>{label_text}:</strong> {score:.2f}</div>",
            unsafe_allow_html=True
        )

# Function to display NER results
def display_ner(ner_result):
    if ner_result:
        st.subheader("Named Entity Recognition (NER) Results:")
        for entity in ner_result:
            st.write(f"Entity: **{entity['word']}**, Label: **{entity['entity']}**, Confidence: {entity['score']:.2f}")
    else:
        st.info("No named entities found.")

# Streamlit app interface
st.title("Text Extraction, Sentiment Analysis and NER")

# File uploader
uploaded_file = st.file_uploader("Upload a .txt, .png, or .jpg file", type=["txt", "png", "jpg"])

if uploaded_file is not None:
    # Extract text from the file
    text = process_file(uploaded_file)
    
    if text:
        st.subheader("Extracted Text:")
        st.write(text)
        
        # Perform sentiment analysis using the Hindi sentiment model
        st.subheader("Sentiment Analysis Result:")
        sentiment_result = sentiment_pipe(text)
        
        # Display sentiment in color-coded boxes
        display_sentiment(sentiment_result)
        
        # Perform Named Entity Recognition (NER)
        ner_result = ner_pipe(text)
        
        # Display NER results
        display_ner(ner_result)
    else:
        st.error("No text could be extracted from the file.")
else:
    st.info("Please upload a file to proceed.")
