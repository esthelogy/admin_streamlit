import streamlit as st
import requests
import os
import re
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
api_base_url = st.secrets["API_BASE_URL"]
base_url = st.secrets["BASE_URL"]

# Initialize Pinecone
try:
    st.write("Initializing Pinecone...")
    pc = Pinecone(api_key=pinecone_api_key)

    # List all indexes
    st.write("Listing Pinecone indexes...")
    all_indexes = pc.list_indexes()
    logging.info(f"Raw Pinecone response: {all_indexes}")

    # Debug: Log the type of all_indexes
    logging.info(f"Type of all_indexes: {type(all_indexes)}")

    # Handle the IndexList object properly
    index_names = []
    if isinstance(all_indexes, list):
        index_names = all_indexes
    elif hasattr(all_indexes, 'indexes'):
        # Access the 'indexes' property directly if it exists
        index_names = [index.name for index in all_indexes.indexes]
    else:
        st.warning(f"Unexpected format of the index list: {all_indexes}")

    logging.info(f"Extracted index names: {index_names}")

    # Check for the specific index
    if index_name not in index_names:
        st.warning(f"Index '{index_name}' not found in Pinecone. Available indexes are: {index_names}")
    else:
        index = pc.Index(index_name)
        logging.info(f"Successfully connected to Pinecone index: {index_name}")
        index_description = index.describe_index_stats()
        logging.info(f"Index dimensions: {index_description['dimension']}")
        logging.info(f"Total vectors: {index_description['total_vector_count']}")

except Exception as e:
    st.error(f"Failed to initialize Pinecone: {str(e)}")
    logging.error(f"Exception encountered: {e}", exc_info=True)
    st.write(f"Error type: {type(e).__name__}")
    st.write(f"Error details: {str(e)}")

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
        logging.info(f"Response content: {response.content}")
        return None
    except Exception as err:
        st.error(f"An error occurred: {err}")
        logging.error(f"Unexpected error: {err}")
        logging.info(f"Response content: {response.content}")
        return None
    logging.info(f"Successful API Response content: {response.content}")
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
        logging.info(f"Response status code: {response.status_code}")
        logging.info(f"Response content: {response.content}")
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
def get_all_quizzes(page: int = 1, size: int = 10):
    url = f"{api_base_url}/quiz/list"
    params = {"page": page, "size": size}
    headers = {"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
    try:
        response = requests.get(url, params=params, headers=headers)
        logging.info(f"Response status code: {response.status_code}")
        logging.info(f"Response content: {response.content}")
        if response.status_code == 200:
            data = response.json()
            return data.get("quizzes", {}).get("items", [])
        elif response.status_code == 404:
            logging.error("Quizzes not found")
        elif response.status_code == 401:
            logging.error("Not authenticated")
        else:
            logging.error(f"Unexpected error: {response.status_code}")
    except Exception as e:
        logging.error(f"Error in get_all_quizzes: {e}")
        st.error(f"An error occurred: {e}")
    return []

# Fetch quiz details by ID
def get_quiz_details(quiz_id: str):
    url = f"{api_base_url}/quiz/{quiz_id}"
    headers = {
        "Authorization": f"Bearer {st.session_state.get('auth_token', '')}",
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, headers=headers)
        logging.info(f"Response status code: {response.status_code}")
        logging.info(f"Response content: {response.content}")
        if response.status_code == 200:
            return response.json().get("quiz", {})
        else:
            st.error(f"Failed to fetch quiz details: {response.status_code}")
            return {}
    except Exception as e:
        logging.error(f"Error in get_quiz_details: {e}")
        st.error(f"An error occurred: {e}")
        return {}

# Update Quiz
def update_quiz(quiz_id: str, quiz_data: dict):
    try:
        response = requests.put(
            f"{api_base_url}/quiz/{quiz_id}",
            json=quiz_data,
            headers={"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
        )
        logging.info(f"Response status code: {response.status_code}")
        logging.info(f"Response content: {response.content}")
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
        logging.info(f"Response status code: {response.status_code}")
        logging.info(f"Response content: {response.content}")
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
        logging.info(f"Response status code: {response.status_code}")
        logging.info(f"Response content: {response.content}")
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
            f"{api_base_url}/admin/esthetician/approval_list",
            params={"page": page, "limit": limit},
            headers={"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
        )
        print(f"{api_base_url}/admin/esthetician/approval_list")
        logging.info(f"Response status code: {response.status_code}")
        logging.info(f"Response content: {response.content}")
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
def approve_esthetician(esthetician_id: str, is_approved: bool = True, reason_for_rejection: str = "N/A") -> str:
    try:
        st.write(f"Reason for rejection: {reason_for_rejection}")
        is_approved_str = "true" if is_approved else "false"
        request_body = {
            "is_approved": is_approved_str,
            "reason_for_rejection": reason_for_rejection if not is_approved else "N/A"
        }
        logging.info(f"Request body for approving esthetician: {request_body}")
        response = requests.put(
            f"{api_base_url}/admin/approve_esthetician/{esthetician_id}",
            json=request_body,
            headers={"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
        )
        logging.info(f"Response status code: {response.status_code}")
        logging.info(f"Response content: {response.content}")
        result = handle_api_response(response)
        if result and result.get("success"):
            status = "approved" if is_approved else "rejected"
            st.success(f"Esthetician {esthetician_id} {status} successfully.")
            return "true"
        else:
            st.error(f"Failed to update esthetician {esthetician_id}.")
            return "false"
    except Exception as e:
        st.error(f"An error occurred while updating esthetician {esthetician_id}.")
        logging.error(f"Error in approve_esthetician: {e}")
        return "false"

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
    size = st.number_input("Quizzes per page", min_value=1, max_value=100, value=10)

    quizzes = get_all_quizzes(page, size)
    if quizzes:
        for quiz in quizzes:
            quiz_id = quiz.get("_id")
            st.write(f"ID: {quiz_id}")
            if quiz_id:
                quiz_details = get_quiz_details(quiz_id)
                title = quiz_details.get("title", "N/A")
                section = quiz_details.get("section", "N/A")
                questions = quiz_details.get("questions", [])

                st.markdown(f"### {title}")
                st.markdown(f"**Section**: {section}")
                # st.write(f"Detail: {quiz_details}")

                for question in questions:
                    st.markdown(f"**Question**: {question.get('question', 'N/A')}")
                    options = question.get("options", [])
                    for option in options:
                        st.markdown(f"- {option}")

                st.markdown("---")

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

    # Check if 'section' key exists in the quiz data and is a string
    if 'section' not in quiz or not isinstance(quiz['section'], str):
        st.error("Quiz data is missing section or section is not a string.")
        return

    # Extract sections from the Pinecone data
    sections = [{"name": quiz['section'], "subsections": [{"name": "Default Subsection", "questions": quiz.get('questions', [])}]}]

    # Edit sections and subsections
    updated_sections = []
    for section in sections:
        st.subheader(f"Section: {section['name']}")
        new_section_name = st.text_input("Section Name", value=section['name'], key=f"section_{section['name']}")

        updated_subsections = []
        for subsection in section['subsections']:
            st.text(f"Subsection: {subsection['name']}")
            new_subsection_name = st.text_input("Subsection Name", value=subsection['name'], key=f"subsection_{subsection['name']}")

            updated_questions = []
            for idx, question in enumerate(subsection['questions']):
                new_question_text = st.text_area("Question", value=question, key=f"question_{idx}")
                if new_question_text != question:
                    updated_questions.append({"id": idx, "text": new_question_text})

            if new_subsection_name != subsection['name'] or updated_questions:
                updated_subsections.append({
                    "name": new_subsection_name,
                    "questions": updated_questions
                })

        if new_section_name != section['name'] or updated_subsections:
            updated_sections.append({
                "name": new_section_name,
                "subsections": updated_subsections
            })

    if st.button("Save Changes"):
        updated_quiz = {
            "id": quiz['_id'],
            "title": new_title,
            "section": updated_sections
        }
        if update_quiz(quiz['_id'], updated_quiz):
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
            st.write(f"ID: {esthetician['_id']}")
            st.write(f"Name: {esthetician['full_name']}, License No: {esthetician['license_no']}")
            st.write(f"Email: {esthetician['email']}, Status: {esthetician['esthetician_status']}")
            st.write(f"License File: {esthetician['license_file']['data']}")
            if esthetician['esthetician_status'] != 'approved':
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Approve {esthetician['full_name']}", key=f"approve_{esthetician['_id']}"):
                        if approve_esthetician(esthetician['_id'], is_approved=True) == "true":
                            st.query_params.update(rerun=True)
                with col2:
                    reason_for_rejection = st.text_input(f"Reason for rejecting {esthetician['full_name']}", key=f"reason_{esthetician['_id']}")
                    if st.button(f"Reject {esthetician['full_name']}", key=f"reject_{esthetician['_id']}"):
                        if approve_esthetician(esthetician['_id'], is_approved=False, reason_for_rejection=reason_for_rejection) == "true":
                            st.query_params.update(rerun=True)
            st.markdown("---")  # Add a horizontal line
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
        logging.info(f"Attempting to login with email: {username}")
        auth_response = authenticate(username, password, api_url)
        logging.info(f"Authentication response: {auth_response}")
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

# Authenticate function
def authenticate(username, password, api_url):
    payload = {"email": username, "password": password}
    try:
        st.write(f"Sending authentication request to: {api_url}")
        response = requests.post(api_url, json=payload)
        st.write(f"Authentication response status code: {response.status_code}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Authentication error: {str(e)}")
        return None

# Define Navigation Menu
def show_navigation_menu():
    st.sidebar.title("Navigation")
    if st.sidebar.button("Admin Page", key="nav_admin_page"):
        st.session_state["page"] = "admin"
    if st.sidebar.button("Esthetician Management", key="nav_esthetician_management"):
        st.session_state["page"] = "esthetician_management"
    if st.sidebar.button("Quiz Management", key="nav_quiz_management"):
        st.session_state["page"] = "quiz_management"
    if st.sidebar.button("Create Quiz", key="nav_create_quiz"):
        st.session_state["page"] = "create_quiz"
    if st.sidebar.button("Logout", key="nav_logout"):
        st.session_state.clear()
        st.session_state["page"] = "login"
        st.success("You have been logged out.")

# Assuming `fetch_quizzes` is a function that fetches quizzes
def fetch_quizzes():
    # This function should return a list of quizzes
    return [
        {'_id': '66c8ab455b1f4a459bb30672', 'title_code': 'basic_information_multiple_choices', 'title': 'Basic Information Multiple Choices', 'section': 'On-boarding Quiz', 'status': 'not_started'},
        # Add other quizzes here
    ]
# Main function to control the app flow
def main():
    #for debugging
    # Sample log entry
    log_entry = "2024-08-25 04:11:52,951 INFO Total quizzes: 0"

    # Define the regular expression pattern
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) (\w+) (.+)'

    # Use re.findall to extract information
    matches = re.findall(pattern, log_entry)

    # Process the extracted information
    for match in matches:
        timestamp, log_level, message = match
        print(f"Timestamp: {timestamp}")
        print(f"Log Level: {log_level}")
        print(f"Message: {message}")
    # Fetch quizzes
    quizzes = fetch_quizzes()

    # Ensure quizzes is a list
    if isinstance(quizzes, list):
        # Method 1: Using len()
        total_quizzes = len(quizzes)
        logging.info(f"Total quizzes using len(): {total_quizzes}")

        # Method 2: Using a loop to count elements
        total_quizzes_loop = 0
        for quiz in quizzes:
            total_quizzes_loop += 1
        logging.info(f"Total quizzes using loop: {total_quizzes_loop}")

        # Method 3: Using sum() with a generator expression
        total_quizzes_sum = sum(1 for _ in quizzes)
        logging.info(f"Total quizzes using sum(): {total_quizzes_sum}")
    else:
        logging.error("Fetched quizzes is not a list.")

    # Original code
    if "page" not in st.session_state:
        st.session_state["page"] = "login"

    show_navigation_menu()  # Display the navigation menu

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
    logging.basicConfig(level=logging.INFO)
    main()
