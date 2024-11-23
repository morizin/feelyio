import streamlit as st
import numpy as np
from PIL import Image
import google.generativeai as genai
import logging
import os
from tempfile import NamedTemporaryFile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Gemini API configuration
def configure_gemini():
    try:
        genai.configure(api_key="AIzaSyC7Y033vd0jZt9hYDZTSa3s6jHmp27-pWg")
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        logging.error(f"Error configuring Gemini API: {e}")
        st.error("Error initializing the AI model. Please check your API configuration.")
        return None

def analyze_emotion_and_refine_text(model, user_text, image_data):
    """
    Analyze the emotion in the image and refine the text accordingly
    """
    prompt = f"""
    1. First, analyze the facial expression and body language in the image to determine the emotional state.
    2. Then, adapt this text to better match and reflect that emotional state: {user_text}
    3. Provide only refined text based on the emotional analysis. (only the text part). 
    Note: there should only be one line of text in the response.
    """
    
    try:
        response = model.generate_content([image_data, prompt])
        logging.info("Successfully generated refined text")
        return response.text
    except Exception as e:
        logging.error(f"Error in text generation: {e}")
        return None

def process_image(img_file):
    """
    Process the captured image and create a temporary file
    """
    try:
        with NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            img = Image.open(img_file)
            img.save(tmp_file.name)
            return genai.upload_file(tmp_file.name)
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return None
    finally:
        if 'tmp_file' in locals():
            os.unlink(tmp_file.name)

def main():
    st.set_page_config(
        page_title="Emotion-Based Text Refinement",
        page_icon="😊",
        layout="centered"
    )

    st.title("✨ Emotion-Based Text Refinement")
    st.write("Transform your text based on your emotional expression!")

    # Initialize Gemini model
    model = configure_gemini()
    if not model:
        return

    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.header("📝 Your Text")
        user_text = st.text_area(
            "Write something:",
            placeholder="Express yourself here...",
            height=150
        )

    with col2:
        st.header("📸 Your Expression")
        img_file_buffer = st.camera_input("Capture your emotion")

    if st.button("✨ Refine My Text", type="primary"):
        # Input validation
        if not user_text.strip():
            st.error("Please enter some text to refine.")
            return
        if not img_file_buffer:
            st.error("Please capture an image of your expression.")
            return

        with st.spinner("Analyzing your emotion and refining text..."):
            # Process image
            image_data = process_image(img_file_buffer)
            if not image_data:
                st.error("Error processing the image. Please try again.")
                return

            # Generate refined text
            refined_result = analyze_emotion_and_refine_text(model, user_text, image_data)
            if refined_result:
                st.success(f"Text refined : {refined_result}")
            else:
                st.error("Failed to refine text. Please try again.")

if __name__ == "__main__":
    main()