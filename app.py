import os

import streamlit as st
from langchain_helper import get_QNA_chain, create_vector_database
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

#  set up the google model.
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash-002')


#  set some configurations.
st.set_page_config(
    page_title="91Trucks Chatbot",
    page_icon=":truck:",
    layout="centered"
)

#  SideBar.
st.sidebar.title("91Trucks WebLinks")
st.sidebar.image("resources/91trucks.jpeg")
#
st.sidebar.link_button(url="https://www.91trucks.com/", label="Visit 91Trucks Website")
st.sidebar.link_button(url="https://www.91tractors.com/", label="Visit 91Tractors Website")
st.sidebar.link_button(url="https://www.91infra.com/", label="Visit 91Infra Website")

st.title("91Trucks:red[.ai]")
# btn = st.button("Create Knowledgebase")

#  Code for creating the vector database.
# question = ""
# if btn:
#     if create_vector_database()[1]:
#         st.success("DataBase Created")
# #

# Initialize chat session in Streamlit if not already present
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

def translate_gemini_user(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return  user_role

# Code for storing the history of the previous chat.
for message in st.session_state.chat_session.history:
    st.chat_message(translate_gemini_user(message.role)).markdown(message.parts[0].text)
#
question = st.chat_input("Ask something about 91Trucks...")
# print("Till Here everything is fine")
if question:
    st.chat_message("user").markdown(question)
    chain = get_QNA_chain()

    # st.chat_message("assistant").markdown(response["result"])
    response = chain(question)
    # Display the assistant's response
    st.chat_message("assistant").markdown(response["result"])
