import streamlit as st
import requests
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

# Function to store question in Pinecone
def store_question(question, embedding):
    try:
        index.upsert(vectors=[{
            "id": question,
            "values": embedding,
            "metadata": {"question": question}
        }])
    except Exception as e:
        st.error("Failed to store the question. Please try again later.")
        logging.error(f"Question storage error: {e}")

# Function to authenticate user
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

# Function to display the questionnaire creation page
def show_questionnaire_page():
    st.title("Create a New Questionnaire")
    st.write("You can add sections, subsections, and questions. Each question will be checked for similarity before being added.")

    if "questionnaire" not in st.session_state:
        st.session_state["questionnaire"] = []

    new_section_name = st.text_input("Enter Section Name")
    new_subsection_name = st.text_input("Enter Subsection Name")

    if st.button("Add Section"):
        if new_section_name:
            st.session_state["questionnaire"].append({"section_name": new_section_name, "subsections": []})
            st.success(f"Section '{new_section_name}' added.")
        else:
            st.error("Please enter a section name.")

    if st.button("Add Subsection"):
        if new_subsection_name and st.session_state["questionnaire"]:
            st.session_state["questionnaire"][-1]["subsections"].append({"subsection_name": new_subsection_name, "questions": []})
            st.success(f"Subsection '{new_subsection_name}' added to Section '{st.session_state['questionnaire'][-1]['section_name']}'.")
        else:
            st.error("Please enter a subsection name or add a section first.")

    for section_idx, section in enumerate(st.session_state["questionnaire"]):
        st.subheader(f"Section {section_idx + 1}: {section['section_name']}")

        if st.button(f"Edit Section {section_idx + 1}"):
            new_name = st.text_input(f"New name for Section {section_idx + 1}", section["section_name"], key=f"edit_section_{section_idx}")
            if st.button(f"Save Section {section_idx + 1}"):
                st.session_state["questionnaire"][section_idx]["section_name"] = new_name
                st.success(f"Section {section_idx + 1} renamed to '{new_name}'.")

        if st.button(f"Delete Section {section_idx + 1}"):
            st.session_state["questionnaire"].pop(section_idx)
            st.success(f"Section {section_idx + 1} deleted.")
            st.experimental_rerun()

        if section_idx > 0 and st.button(f"Move Section {section_idx + 1} Up"):
            st.session_state["questionnaire"].insert(section_idx - 1, st.session_state["questionnaire"].pop(section_idx))
            st.experimental_rerun()

        for subsection_idx, subsection in enumerate(section["subsections"]):
            st.text(f"Subsection {subsection_idx + 1}: {subsection['subsection_name']}")

            if st.button(f"Edit Subsection {section_idx + 1}.{subsection_idx + 1}"):
                new_name = st.text_input(f"New name for Subsection {section_idx + 1}.{subsection_idx + 1}", subsection["subsection_name"], key=f"edit_subsection_{section_idx}_{subsection_idx}")
                if st.button(f"Save Subsection {section_idx + 1}.{subsection_idx + 1}"):
                    st.session_state["questionnaire"][section_idx]["subsections"][subsection_idx]["subsection_name"] = new_name
                    st.success(f"Subsection {section_idx + 1}.{subsection_idx + 1} renamed to '{new_name}'.")

            if st.button(f"Delete Subsection {section_idx + 1}.{subsection_idx + 1}"):
                st.session_state["questionnaire"][section_idx]["subsections"].pop(subsection_idx)
                st.success(f"Subsection {section_idx + 1}.{subsection_idx + 1} deleted.")
                st.experimental_rerun()

            if subsection_idx > 0 and st.button(f"Move Subsection {section_idx + 1}.{subsection_idx + 1} Up"):
                st.session_state["questionnaire"][section_idx]["subsections"].insert(subsection_idx - 1, st.session_state["questionnaire"][section_idx]["subsections"].pop(subsection_idx))
                st.experimental_rerun()

            for question_idx, question in enumerate(subsection["questions"]):
                st.text(f"Question {question_idx + 1}: {question}")

                if st.button(f"Edit Question {section_idx + 1}.{subsection_idx + 1}.{question_idx + 1}"):
                    new_question = st.text_input(f"New text for Question {section_idx + 1}.{subsection_idx + 1}.{question_idx + 1}", question, key=f"edit_question_{section_idx}_{subsection_idx}_{question_idx}")
                    if st.button(f"Save Question {section_idx + 1}.{subsection_idx + 1}.{question_idx + 1}"):
                        st.session_state["questionnaire"][section_idx]["subsections"][subsection_idx]["questions"][question_idx] = new_question
                        st.success(f"Question {section_idx + 1}.{subsection_idx + 1}.{question_idx + 1} updated.")

                if st.button(f"Delete Question {section_idx + 1}.{subsection_idx + 1}.{question_idx + 1}"):
                    st.session_state["questionnaire"][section_idx]["subsections"][subsection_idx]["questions"].pop(question_idx)
                    st.success(f"Question {section_idx + 1}.{subsection_idx + 1}.{question_idx + 1} deleted.")
                    st.experimental_rerun()

                if question_idx > 0 and st.button(f"Move Question {section_idx + 1}.{subsection_idx + 1}.{question_idx + 1} Up"):
                    st.session_state["questionnaire"][section_idx]["subsections"][subsection_idx]["questions"].insert(question_idx - 1, st.session_state["questionnaire"][section_idx]["subsections"][subsection_idx]["questions"].pop(question_idx))
                    st.experimental_rerun()

    if st.button("Save Questionnaire"):
        st.success("Questionnaire saved successfully!")
        st.write("Saved Questionnaire Data:")
        st.json(st.session_state["questionnaire"])

    if st.button("Back"):
        st.session_state["page"] = "login"

# Function to display the login page
def show_login_page():
    st.title("Esthelogy Admin")

    username = st.text_input("Email")
    password = st.text_input("Password", type="password")
    api_url = "https://dev-eciabackend.esthelogy.com/esthelogy/v1.0/user/login"

    login_button_clicked = st.button("Login")
    if login_button_clicked:
        auth_response = authenticate(username, password, api_url)

        if auth_response:
            if auth_response.get("success") and auth_response.get("role") == "admin":
                st.success("Login successful!")
                st.session_state["auth_token"] = auth_response.get("access_token", "")
                st.session_state["user_id"] = auth_response.get("user_id", "")
                st.session_state["page"] = "questionnaire"
                st.write("Redirecting to admin page... Click on Login button again")
            elif auth_response.get("role") != "admin":
                st.error("Access denied: You do not have admin privileges.")
            else:
                st.error(f"Login failed: {auth_response.get('message')}")
        else:
            st.error("Login failed. Please check your credentials or API URL.")

# Main function to control the flow
def main():
    if "page" not in st.session_state:
        st.session_state["page"] = "login"

    if st.session_state["page"] == "login":
        show_login_page()
    elif st.session_state["page"] == "questionnaire":
        show_questionnaire_page()

if __name__ == "__main__":
    main()
