import streamlit as st
import requests
import os
import logging
from openai import OpenAI
from pinecone import Pinecone
from functools import lru_cache

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Load API keys from Streamlit secrets
try:
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except KeyError as e:
    st.error("API keys are missing. Please configure them in the Streamlit app settings under Secrets.")
    logging.error(f"Missing API key: {e}")
    st.stop()

# Initialize Pinecone and OpenAI
try:
    pc = Pinecone(api_key=pinecone_api_key)
    client = OpenAI(api_key=openai_api_key)
    index_name = "questionnaire-index"
    index = pc.Index(index_name)
except Exception as e:
    st.error("Failed to initialize Pinecone or OpenAI. Please check your API keys.")
    logging.error(f"Initialization error: {e}")
    st.stop()

# Caching for embedding generation to improve scalability
@lru_cache(maxsize=1024)
def get_embedding(text, model="text-embedding-3-small"):
    try:
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        st.error("Failed to generate embedding. Please try again later.")
        logging.error(f"Embedding generation error: {e}")
        return None

# Function to check for similar questions in Pinecone
def check_similarity(embedding, threshold=0.6):
    try:
        query_result = index.query(vector=embedding, top_k=1, include_metadata=True)
        if query_result['matches']:
            score = query_result['matches'][0]['score']
            if score > threshold:
                return query_result['matches'][0]['metadata']['question'], score
    except Exception as e:
        st.error("Failed to check similarity. Please try again later.")
        logging.error(f"Similarity check error: {e}")
    return None, None

# API URL
api_base_url = "https://dev-eciabackend.esthelogy.com/esthelogy/v1.0"

# Helper function to handle API responses
def handle_api_response(response):
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {response.json().get('message', str(http_err))}")
        logging.error(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        st.error(f"An error occurred: {err}")
        logging.error(f"Unexpected error: {err}")
        return None
    return response.json()

# Start Quiz
def start_quiz(section_code):
    try:
        response = requests.get(f"{api_base_url}/quiz/start_quiz", params={"section_code": section_code})
        return handle_api_response(response)
    except Exception as e:
        st.error("Failed to start quiz. Please try again later.")
        logging.error(f"Error in start_quiz: {e}")
        return None

# Fetch Next Question
def fetch_next_question(quiz_id, question_id, response, response_time):
    payload = {
        "quiz_id": quiz_id,
        "question_id": question_id,
        "response": response,
        "response_time": response_time
    }
    try:
        response = requests.post(f"{api_base_url}/quiz/fetch_next", json=payload)
        return handle_api_response(response)
    except Exception as e:
        st.error("Failed to fetch next question. Please try again later.")
        logging.error(f"Error in fetch_next_question: {e}")
        return None

# Submit Full Quiz
def submit_full_quiz(quiz_id):
    payload = {"quiz_id": quiz_id}
    try:
        response = requests.post(f"{api_base_url}/quiz/submit", json=payload)
        return handle_api_response(response)
    except Exception as e:
        st.error("Failed to submit quiz. Please try again later.")
        logging.error(f"Error in submit_full_quiz: {e}")
        return None

# Main Admin Page after login
def show_admin_page():
    st.title("Admin Page")

    # Display main menu
    menu = ["Approval Management", "Quiz Management"]
    choice = st.sidebar.selectbox("Select an option", menu)

    if choice == "Approval Management":
        st.subheader("Approval Management")
        st.write("This function will be implemented later.")

    elif choice == "Quiz Management":
        st.subheader("Quiz Management")

        # List all quizzes (Assuming an API endpoint exists to get all quizzes)
        quizzes = get_all_quizzes()
        if quizzes:
            for quiz in quizzes:
                st.write(f"Quiz ID: {quiz['quiz_id']}, Section: {quiz['section_code']}")
                if st.button(f"Edit Quiz {quiz['quiz_id']}", key=f"edit_{quiz['quiz_id']}"):
                    edit_quiz(quiz['quiz_id'])
                if st.button(f"Delete Quiz {quiz['quiz_id']}", key=f"delete_{quiz['quiz_id']}"):
                    delete_quiz(quiz['quiz_id'])

        if st.button("Create New Quiz"):
            create_quiz()

# Fetch all quizzes (Example, assuming this API exists)
def get_all_quizzes():
    try:
        response = requests.get(f"{api_base_url}/quiz/list")
        return handle_api_response(response)
    except Exception as e:
        st.error("Failed to fetch quizzes. Please try again later.")
        logging.error(f"Error in get_all_quizzes: {e}")
        return None

# Edit quiz function (To be implemented)
def edit_quiz(quiz_id):
    st.write(f"Editing Quiz {quiz_id}...")
    # Implement the functionality for editing the quiz

# Delete quiz function
def delete_quiz(quiz_id):
    try:
        response = requests.delete(f"{api_base_url}/quiz/{quiz_id}")
        result = handle_api_response(response)
        if result and result.get("success"):
            st.success(f"Quiz {quiz_id} deleted successfully!")
        else:
            st.error(f"Failed to delete quiz {quiz_id}.")
    except Exception as e:
        st.error(f"Failed to delete quiz {quiz_id}. Please try again later.")
        logging.error(f"Error in delete_quiz: {e}")

# Create new quiz function (To be implemented)
def create_quiz():
    st.write("Creating a new quiz...")
    # Implement the functionality for creating a new quiz

# Function to display the login page
def show_login_page():
    st.title("Esthelogy Admin")

    username = st.text_input("Email")
    password = st.text_input("Password", type="password")
    api_url = "https://dev-eciabackend.esthelogy.com/esthelogy/v1.0/user/login"

    if st.button("Login"):
        auth_response = authenticate(username, password, api_url)

        if auth_response:
            if auth_response.get("success") and auth_response.get("role") == "admin":
                st.success("Login successful!")
                st.session_state["auth_token"] = auth_response.get("access_token", "")
                st.session_state["user_id"] = auth_response.get("user_id", "")
                st.session_state["page"] = "admin"
            elif auth_response.get("role") != "admin":
                st.error("Access denied: You do not have admin privileges.")
            else:
                st.error(f"Login failed: {auth_response.get('message')}")
        else:
            st.error("Login failed. Please check your credentials or API URL.")

# Authenticate user
def authenticate(username, password, api_url):
    payload = {"email": username, "password": password}
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error("Failed to authenticate. Please check your credentials or API URL.")
        logging.error(f"Authentication error: {e}")
        return None

# Main function to control the app flow
def main():
    if "page" not in st.session_state:
        st.session_state["page"] = "login"

    if st.session_state["page"] == "login":
        show_login_page()
    elif st.session_state["page"] == "admin":
        show_admin_page()

if __name__ == "__main__":
    main()
