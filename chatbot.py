import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
# #  Load the env files.
load_dotenv()
#
# # # Initiallize the api key.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# #
# # #  Setup some streamlit configuration.
st.set_page_config(
    page_title="91Trucks",
    page_icon=":truck:",
    layout="centered"
)
# #
# # #  Set up the llm gemini model.
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash-002")
#
# # Initialize chat session in Streamlit if not already present
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])
#
# # Function to display the message according to the role.
def translate_gemini_user(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return  user_role
#
# # UI Display.
st.title("🤖 91Trucks Chatbot")
#
# #  Code for display the chat history.
for message in st.session_state.chat_session.history:
    st.chat_message(translate_gemini_user(message.role)).markdown(message.parts[0].text)
    # print(message)
#
# # Create the streamlit ui for the chatbot.
user_prompt = st.chat_input("Ask something about 91Trucks...")
if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    # now generate the gemini response.
    gemini_response = st.session_state.chat_session.send_message(user_prompt)
    # gemini_response = model.generate_content([user_prompt])               # By doing this we are not able to save the history

    #  Display the gemini response into the chatmessage.
    st.chat_message("assistant").markdown(gemini_response.text)