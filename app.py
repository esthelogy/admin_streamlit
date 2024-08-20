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

# Main Quiz Function
def run_quiz():
    st.title("Quiz Time!")

    if "quiz_id" not in st.session_state:
        st.session_state.quiz_id = None
        st.session_state.current_question = None
        st.session_state.current_question_index = 0

    # Start quiz
    section_code = st.text_input("Enter the section code to start the quiz")
    if st.button("Start Quiz"):
        if section_code:
            result = start_quiz(section_code)
            if result and result.get("success"):
                st.session_state.quiz_id = result["question_details"]["quiz_id"]
                st.session_state.current_question = result["question_details"]["question"]
                st.session_state.total_question_count = result["question_details"]["total_question_count"]
                st.session_state.current_question_index = result["question_details"]["current_question_index"]
                st.success("Quiz started successfully!")
            else:
                st.error(result.get("message", "Failed to start quiz"))

    # Display current question
    if st.session_state.quiz_id and st.session_state.current_question:
        question_data = st.session_state.current_question
        st.subheader(f"Question {st.session_state.current_question_index + 1}")
        st.write(question_data["question"])

        # Display options
        selected_option = st.radio("Select an option:", question_data["options"])

        if st.button("Submit and Next"):
            result = fetch_next_question(
                quiz_id=st.session_state.quiz_id,
                question_id=question_data["question_id"],
                response=selected_option,
                response_time=0  # In a real scenario, you would measure the time taken to respond
            )
            if result and result.get("success"):
                if st.session_state.current_question_index + 1 < st.session_state.total_question_count:
                    st.session_state.current_question = result["question_details"]["question"]
                    st.session_state.current_question_index += 1
                    st.success("Next question loaded.")
                else:
                    st.session_state.current_question = None
                    st.success("Quiz completed! Submitting your answers...")
                    submit_full_quiz(st.session_state.quiz_id)
            else:
                st.error(result.get("message", "Failed to fetch the next question"))

    # Finalize the quiz
    if st.session_state.quiz_id and st.session_state.current_question is None:
        st.success("You have completed the quiz! Thank you for participating.")
        if st.button("Submit Quiz"):
            result = submit_full_quiz(st.session_state.quiz_id)
            if result and result.get("success"):
                st.success("Quiz submitted successfully!")
                st.session_state.clear()
            else:
                st.error(result.get("message", "Failed to submit the quiz"))

# Main function to control the app flow
def main():
    run_quiz()

if __name__ == "__main__":
    main()
