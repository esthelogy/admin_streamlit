import streamlit as st
import requests
import os
import logging
from openai import OpenAI
from pinecone import Pinecone
from functools import lru_cache
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Configure logging
#logging.basicConfig(level=logging.DEBUG)

# Load API keys from Streamlit secrets
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
except KeyError as e:
    st.error(f"API key is missing: {e}. Please configure it in the Streamlit app settings under Secrets.")
    logging.error(f"Missing API key: {e}")
    st.stop()

# Hard-coded index name
index_name = "title-index"

# API base URL
api_base_url = "https://dev-eciabackend.esthelogy.com/esthelogy/v1.0"

# Pinecone initialization
st.write("Pinecone initialization started")
try:
    logging.debug("Starting Pinecone initialization")
    pc = Pinecone(api_key=pinecone_api_key)
    logging.debug("Pinecone client created")
    
    all_indexes = pc.list_indexes()
    logging.debug(f"All indexes: {all_indexes}")
    
    if isinstance(all_indexes, dict) and 'indexes' in all_indexes:
        index_names = [index['name'] for index in all_indexes['indexes']]
    else:
        index_names = all_indexes if isinstance(all_indexes, list) else []
    logging.debug(f"Extracted index names: {index_names}")
    
    if index_name not in index_names:
        logging.warning(f"Index '{index_name}' not found. Available indexes: {index_names}")
    else:
        index = pc.Index(index_name)
        logging.debug(f"Successfully connected to index: {index_name}")
        index_stats = index.describe_index_stats()
        logging.debug(f"Index stats: {index_stats}")

except Exception as e:
    logging.error(f"Error during Pinecone initialization: {str(e)}", exc_info=True)
    st.error(f"Failed to initialize Pinecone: {str(e)}")
st.write("App initialization completed")
logging.debug("Pinecone initialization complete")

# Initialize OpenAI (keep this part as is)
try:
    client = OpenAI(api_key=openai_api_key)
except Exception as e:
    st.error(f"Failed to initialize OpenAI: {str(e)}")
    logging.error(f"OpenAI initialization error: {e}")
    st.stop()

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
    if 'index' not in globals():
        logging.warning("Pinecone index not available. Similarity check skipped.")
        return None, None
    try:
        query_result = index.query(vector=embedding, top_k=1, include_metadata=True)
        if query_result['matches']:
            score = query_result['matches'][0]['score']
            if score > threshold:
                return query_result['matches'][0]['metadata']['question'], score
    except Exception as e:
        logging.error(f"Similarity check error: {e}")
    return None, None
    
# Create Quiz
def create_quiz(quiz_data):
    try:
        response = requests.post(
            f"{api_base_url}/quiz/create",
            json=quiz_data,
            headers={"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
        )
        result = handle_api_response(response)
        if result and result.get("success"):
            st.success("Quiz created successfully!")
            return result.get("quiz")
        else:
            st.error(f"Failed to create quiz: {result.get('message', 'Unknown error')}")
    except Exception as e:
        st.error("Failed to create quiz. Please try again later.")
        logging.error(f"Error in create_quiz: {e}")
    return None

# Fetch All Quizzes
def get_all_quizzes(page: int = 1, limit: int = 10):
    try:
        response = requests.get(
            f"{api_base_url}/quiz/list",
            params={"page": page, "limit": limit},
            headers={"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
        )
        result = handle_api_response(response)
        if result and result.get("success"):
            return result.get("quizzes", []), result.get("total", 0)
        else:
            st.error("Failed to fetch quizzes. Please try again later.")
    except Exception as e:
        st.error("Failed to fetch quizzes. Please try again later.")
        logging.error(f"Error in get_all_quizzes: {e}")
    return [], 0

# Update Quiz
def update_quiz(quiz_id: str, quiz_data: dict):
    try:
        response = requests.put(
            f"{api_base_url}/quiz/{quiz_id}",
            json=quiz_data,
            headers={"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
        )
        result = handle_api_response(response)
        if result and result.get("success"):
            st.success(f"Quiz {quiz_id} updated successfully!")
            return result.get("quiz")
        else:
            st.error(f"Failed to update quiz {quiz_id}: {result.get('message', 'Unknown error')}")
    except Exception as e:
        st.error(f"Failed to update quiz {quiz_id}. Please try again later.")
        logging.error(f"Error in update_quiz: {e}")
    return None

# Delete Quiz
def delete_quiz(quiz_id: str):
    try:
        response = requests.delete(
            f"{api_base_url}/quiz/{quiz_id}",
            headers={"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
        )
        result = handle_api_response(response)
        if result and result.get("success"):
            st.success(f"Quiz {quiz_id} deleted successfully!")
            return True
        else:
            st.error(f"Failed to delete quiz {quiz_id}: {result.get('message', 'Unknown error')}")
    except Exception as e:
        st.error(f"Failed to delete quiz {quiz_id}. Please try again later.")
        logging.error(f"Error in delete_quiz: {e}")
    return False

# Upload Puzzle
def upload_puzzle(file, quiz_id: str):
    try:
        files = {"file": file}
        data = {"quiz_id": quiz_id}
        response = requests.post(
            f"{api_base_url}/admin/quiz/puzzle/upload",
            files=files,
            data=data,
            headers={"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
        )
        result = handle_api_response(response)
        if result and result.get("success"):
            st.success("Puzzle uploaded successfully!")
            return result.get("puzzle")
        else:
            st.error(f"Failed to upload puzzle: {result.get('message', 'Unknown error')}")
    except Exception as e:
        st.error("Failed to upload puzzle. Please try again later.")
        logging.error(f"Error in upload_puzzle: {e}")
    return None

# List Estheticians
def list_estheticians(page: int = 1, limit: int = 10) -> List[Dict[str, Any]]:
    try:
        response = requests.get(
            f"{api_base_url}/admin/esthetician/list",
            params={"page": page, "limit": limit},
            headers={"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
        )
        result = handle_api_response(response)
        if result and result.get("success"):
            return result.get("estheticians", [])
        else:
            st.error("Failed to fetch estheticians. Please try again later.")
            return []
    except Exception as e:
        st.error("An error occurred while fetching estheticians.")
        logging.error(f"Error in list_estheticians: {e}")
        return []

# Approve Esthetician
def approve_esthetician(esthetician_id: str) -> bool:
    try:
        response = requests.get(
            f"{api_base_url}/admin/approve/esthetician/{esthetician_id}",
            headers={"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
        )
        result = handle_api_response(response)
        if result and result.get("success"):
            st.success(f"Esthetician {esthetician_id} approved successfully.")
            return True
        else:
            st.error(f"Failed to approve esthetician {esthetician_id}.")
            return False
    except Exception as e:
        st.error(f"An error occurred while approving esthetician {esthetician_id}.")
        logging.error(f"Error in approve_esthetician: {e}")
        return False

# Show Admin Page
def show_admin_page():
    st.title("Admin Page")
    st.subheader("Menu")
    
    if st.button("Esthetician Management"):
        st.session_state["page"] = "esthetician_management"
    
    if st.button("Quiz Management"):
        st.session_state["page"] = "quiz_management"

    if st.button("Logout"):
        st.session_state.clear()
        st.session_state["page"] = "login"
        st.success("You have been logged out.")

# Show Quiz Management Page
def show_quiz_management():
    st.subheader("Quiz Management")

    page = st.number_input("Page", min_value=1, value=1)
    limit = st.number_input("Quizzes per page", min_value=1, max_value=100, value=10)

    quizzes, total_quizzes = get_all_quizzes(page, limit)
    st.write(f"Total Quizzes: {total_quizzes}")

    if quizzes:
        for quiz in quizzes:
            st.write(f"Quiz ID: {quiz['id']}, Title: {quiz['title']}")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"Edit Quiz {quiz['id']}", key=f"edit_{quiz['id']}"):
                    st.session_state["editing_quiz"] = quiz
                    st.session_state["page"] = "edit_quiz"
            with col2:
                if st.button(f"Delete Quiz {quiz['id']}", key=f"delete_{quiz['id']}"):
                    if delete_quiz(quiz['id']):
                        st.experimental_rerun()
            with col3:
                uploaded_file = st.file_uploader(f"Upload Puzzle for Quiz {quiz['id']}", key=f"upload_{quiz['id']}")
                if uploaded_file is not None:
                    if upload_puzzle(uploaded_file, quiz['id']):
                        st.experimental_rerun()

    if st.button("Create New Quiz"):
        st.session_state["page"] = "create_quiz"

    if st.button("Back to Admin Page"):
        st.session_state["page"] = "admin"

# Create Quiz Page
def create_quiz_page():
    st.title("Create a New Quiz")
    st.write("You can add sections, subsections, and questions. Each question will be checked for similarity before being added.")

    if "new_quiz" not in st.session_state:
        st.session_state["new_quiz"] = {
            "quiz_title": "",
            "sections": []
        }

    st.session_state["new_quiz"]["quiz_title"] = st.text_input("Enter Quiz Title", value=st.session_state["new_quiz"]["quiz_title"], help="The title of the quiz.")
    new_section_name = st.text_input("Enter Section Name", help="Add a new section to your quiz.")
    new_subsection_name = st.text_input("Enter Subsection Name", help="Add a new subsection under the section.")

    if st.button("Add Section"):
        if new_section_name:
            st.session_state["new_quiz"]["sections"].append({
                "section_name": new_section_name,
                "subsections": []
            })
            st.success(f"Section '{new_section_name}' added.")
        else:
            st.error("Please enter a section name.")

    if st.button("Add Subsection"):
        if new_subsection_name and st.session_state["new_quiz"]["sections"]:
            st.session_state["new_quiz"]["sections"][-1]["subsections"].append({
                "subsection_name": new_subsection_name,
                "questions": []
            })
            st.success(f"Subsection '{new_subsection_name}' added to Section '{st.session_state['new_quiz']['sections'][-1]['section_name']}'.")
        else:
            st.error("Please enter a subsection name or add a section first.")

    for section_idx, section in enumerate(st.session_state["new_quiz"]["sections"]):
        st.subheader(f"Section {section_idx + 1}: {section['section_name']}")
        for subsection_idx, subsection in enumerate(section["subsections"]):
            st.text(f"Subsection {subsection_idx + 1}: {subsection['subsection_name']}")

            question_text = st.text_input(f"Enter a question for {subsection['subsection_name']}:", key=f"question_{section_idx}_{subsection_idx}")
            if st.button(f"Add Question to {subsection['subsection_name']}", key=f"add_question_{section_idx}_{subsection_idx}"):
                if question_text:
                    embedding = get_embedding(question_text)
                    similar_question, score = check_similarity(embedding)
                    if similar_question:
                        st.warning(f"This question is similar to an existing question: '{similar_question}' with a similarity score of {(score*100):.2f}. Consider revising it.")
                    else:
                        subsection["questions"].append(question_text)
                        st.success(f"Question added to {subsection['subsection_name']}.")
                else:
                    st.error("Please enter a question.")

    if st.button("Save Quiz"):
        if create_quiz(st.session_state["new_quiz"]):
            st.success("Quiz saved successfully!")
            st.session_state.pop("new_quiz")
        else:
            st.error("Failed to save the quiz. Please try again later.")

    if st.button("Back"):
        st.session_state["page"] = "quiz_management"

# Edit Quiz Page
def edit_quiz_page():
    if "editing_quiz" not in st.session_state:
        st.error("No quiz selected for editing.")
        st.session_state["page"] = "quiz_management"
        return

    quiz = st.session_state["editing_quiz"]
    st.title(f"Edit Quiz: {quiz['title']}")

    # Edit quiz title
    new_title = st.text_input("Quiz Title", value=quiz['title'])

    # Edit sections and subsections
    updated_sections = []
    for section in quiz['sections']:
        st.subheader(f"Section: {section['name']}")
        new_section_name = st.text_input("Section Name", value=section['name'], key=f"section_{section['id']}")
        
        updated_subsections = []
        for subsection in section['subsections']:
            st.text(f"Subsection: {subsection['name']}")
            new_subsection_name = st.text_input("Subsection Name", value=subsection['name'], key=f"subsection_{subsection['id']}")
            
            updated_questions = []
            for question in subsection['questions']:
                new_question_text = st.text_area("Question", value=question['text'], key=f"question_{question['id']}")
                if new_question_text != question['text']:
                    updated_questions.append({"id": question['id'], "text": new_question_text})
            
            if new_subsection_name != subsection['name'] or updated_questions:
                updated_subsections.append({
                    "id": subsection['id'],
                    "name": new_subsection_name,
                    "questions": updated_questions
                })
        
        if new_section_name != section['name'] or updated_subsections:
            updated_sections.append({
                "id": section['id'],
                "name": new_section_name,
                "subsections": updated_subsections
            })

    if st.button("Save Changes"):
        updated_quiz = {
            "id": quiz['id'],
            "title": new_title,
            "sections": updated_sections
        }
        if update_quiz(quiz['id'], updated_quiz):
            st.success("Quiz updated successfully!")
            st.session_state.pop("editing_quiz")
            st.session_state["page"] = "quiz_management"
        else:
            st.error("Failed to update quiz. Please try again.")

    if st.button("Cancel"):
        st.session_state.pop("editing_quiz")
        st.session_state["page"] = "quiz_management"

# Show Esthetician Management
def show_esthetician_management():
    st.title("Esthetician Management")

    page = st.number_input("Page", min_value=1, value=1)
    limit = st.number_input("Estheticians per page", min_value=1, max_value=100, value=10)

    estheticians = list_estheticians(page, limit)
    
    if estheticians:
        for esthetician in estheticians:
            st.write(f"ID: {esthetician['id']}, Name: {esthetician['name']}, Email: {esthetician['email']}")
            if esthetician['status'] != 'approved':
                if st.button(f"Approve {esthetician['name']}", key=f"approve_{esthetician['id']}"):
                    if approve_esthetician(esthetician['id']):
                        st.experimental_rerun()
    else:
        st.write("No estheticians found or failed to fetch the list.")

    if st.button("Back to Admin Page"):
        st.session_state["page"] = "admin"

# Show Login Page
def show_login_page():
    st.title("Esthelogy Admin")

    username = st.text_input("Email")
    password = st.text_input("Password", type="password")
    api_url = f"{api_base_url}/user/login"

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
    elif st.session_state["page"] == "quiz_management":
        show_quiz_management()
    elif st.session_state["page"] == "create_quiz":
        create_quiz_page()
    elif st.session_state["page"] == "edit_quiz":
        edit_quiz_page()
    elif st.session_state["page"] == "esthetician_management":
        show_esthetician_management()

if __name__ == "__main__":
    main()
