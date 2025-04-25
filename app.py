import streamlit as st
import openai
from gtts import gTTS
import io
import base64
import tempfile
from streamlit_mic_recorder import mic_recorder

# Initialize OpenAI client with API key from secrets
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Speak to me, and I'll respond."}]

# Streamlit page configuration
st.set_page_config(page_title="Voice Chatbot", page_icon="🎤")
st.title("Voice Chatbot with OpenAI")

# Function to convert speech to text using OpenAI Whisper
def speech_to_text():
    audio = mic_recorder(start_prompt="🎙️ Record", stop_prompt="🛑 Stop", key="recorder")
    if audio:
        try:
            # Save audio to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio.write(audio['bytes'])
                temp_audio_path = temp_audio.name

            # Transcribe using OpenAI Whisper
            with open(temp_audio_path, "rb") as audio_file:
                transcription = openai.Audio.transcribe(model="whisper-1", file=audio_file)
            
            # Clean up temporary file
            import os
            os.unlink(temp_audio_path)
            
            return transcription['text']
        except Exception as e:
            return f"Error in speech recognition: {str(e)}"
    return None

# Function to get ChatGPT response
def get_chatgpt_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error communicating with OpenAI API: {str(e)}"

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    audio_bytes = audio_buffer.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    return audio_base64

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "audio" in message:
            st.audio(f"data:audio/mp3;base64,{message['audio']}", format="audio/mp3")

# Handle voice input
prompt = speech_to_text()
if prompt:
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get and display assistant response
    response = get_chatgpt_response(prompt)
    audio_base64 = text_to_speech(response)
    
    with st.chat_message("assistant"):
        st.markdown(response)
        st.audio(f"data:audio/mp3;base64,{audio_base64}", format="audio/mp3")
    
    st.session_state.messages.append({"role": "assistant", "content": response, "audio": audio_base64})

# Instructions for user
st.write("Click the microphone to record your voice. Speak clearly, and the assistant will respond with voice output.")