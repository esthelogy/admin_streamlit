import streamlit as st
import requests
from openai import OpenAI
from pinecone import Pinecone

# Initialize Pinecone
pc = Pinecone(api_key="76b51c9a-5d43-41df-93cb-08fb39db0f48")

index_name = "questionnaire-index"
# Connect to the index
index = pc.Index(index_name)

# Initialize OpenAI
client = OpenAI(api_key='sk-svcacct-DVxt9NzwdeX4dl72jrLGZIICVo9-vxAB1zBUb5TWioOP7_7DPT3BlbkFJiok0wQ4tJVPvZVlTDU33RDI1QuiL8DyAVI9Doas5t2yOVNKFgA')

# Function to get embeddings from OpenAI
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


# Function to check for similar questions in Pinecone
def check_similarity(embedding, threshold=0.6):
    query_result = index.query(vector=embedding, top_k=1, include_metadata=True)
    if query_result['matches']:
        score = query_result['matches'][0]['score']
        if score > threshold:
            return query_result['matches'][0]['metadata']['question'], score
    return None, None

# Function to store question in Pinecone
def store_question(question, embedding):
    index.upsert(vectors=[{
        "id": question,
        "values": embedding,
        "metadata": {"question": question}
    }])

# Function to authenticate user
def authenticate(username, password, api_url):
    payload = {"email": username, "password": password}
    response = requests.post(api_url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Function to display the questionnaire creation page
def show_questionnaire_page():
    st.title("Create a New Questionnaire")
    st.write("You can add sections, subsections, and questions. Each question will be checked for similarity before being added.")

    # Initialize session state to store the questionnaire structure if not already done
    if "questionnaire" not in st.session_state:
        st.session_state["questionnaire"] = []

    # Input for new section name
    new_section_name = st.text_input("Enter Section Name", help="Add a new section to your questionnaire.")

    # Input for new subsection name
    new_subsection_name = st.text_input("Enter Subsection Name", help="Add a new subsection under the section.")

    # Add a new section when the button is clicked
    if st.button("Add Section"):
        if new_section_name:
            st.session_state["questionnaire"].append({
                "section_name": new_section_name,
                "subsections": []
            })
            st.success(f"Section '{new_section_name}' added.")
        else:
            st.error("Please enter a section name.")

    # Add a new subsection to the last section
    if st.button("Add Subsection"):
        if new_subsection_name and st.session_state["questionnaire"]:
            st.session_state["questionnaire"][-1]["subsections"].append({
                "subsection_name": new_subsection_name,
                "questions": []
            })
            st.success(f"Subsection '{new_subsection_name}' added to Section '{st.session_state['questionnaire'][-1]['section_name']}'.")
        else:
            st.error("Please enter a subsection name or add a section first.")

    # Display existing sections and subsections
    for section_idx, section in enumerate(st.session_state["questionnaire"]):
        st.subheader(f"Section {section_idx + 1}: {section['section_name']}")

        for subsection_idx, subsection in enumerate(section["subsections"]):
            st.text(f"Subsection {subsection_idx + 1}: {subsection['subsection_name']}")

            # Add questions to the subsection
            question_text = st.text_input(f"Enter a question for {subsection['subsection_name']}:", key=f"question_{section_idx}_{subsection_idx}")
            if st.button(f"Add Question to {subsection['subsection_name']}", key=f"add_question_{section_idx}_{subsection_idx}"):
                if question_text:
                    # Get the embedding for the question
                    embedding = get_embedding(question_text)
                    
                    # Check for similar questions in Pinecone
                    similar_question, score = check_similarity(embedding)
                    if similar_question:
                        st.warning(f"This question is similar to an existing question: '{similar_question}' with a similarity score of {(score*100):.2f}. Consider revising it.")
                    else:
                        subsection["questions"].append(question_text)
                        store_question(question_text, embedding)
                        st.success(f"Question added to {subsection['subsection_name']}.")
                else:
                    st.error("Please enter a question.")

    # Save the questionnaire
    if st.button("Save Questionnaire"):
        st.success("Questionnaire saved successfully!")
        st.write("Saved Questionnaire Data:")
        st.json(st.session_state["questionnaire"])

    # Back button
    if st.button("Back"):
        st.session_state["page"] = "login"

# Function to display the login page
def show_login_page():
    st.title("Esthelogy Admin")

    # Input fields for username and password
    username = st.text_input("Email")
    password = st.text_input("Password", type="password")
    api_url = "https://dev-eciabackend.esthelogy.com/esthelogy/v1.0/user/login"

    # Button to login
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
