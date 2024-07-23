import streamlit as st
from utils import (
    extract_text_from_pdf,
    summarize_text,
    format_index_name
)
from agent_utils import createLearningAgent, createNewVectorIndex, getExistingVectorStore, checkIfNewIndex
from dotenv import load_dotenv
import os
from langchain.memory import ConversationBufferWindowMemory

# Load environment variables
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["PINECONE_API_KEY"] = pinecone_api_key

# Set up Streamlit UI
if 'index_name' not in st.session_state:
    st.session_state['index_name'] = None
if 'topics' not in st.session_state:
    st.session_state['topics'] = None
if 'assistant' not in st.session_state:
    st.session_state['assistant'] = None
if 'started' not in st.session_state:
    st.session_state['started'] = False
if 'doc_messages' not in st.session_state:
    # Start with first message from assistant
    st.session_state['doc_messages'] = []
st.sidebar.write("Instructions")
st.sidebar.info("1. Create lesson first.\n2. Modify or add user info\n3. Click to start the lesson\n4. Start learning")
# memory = ConversationBufferWindowMemory(memory_key = "chat_history", return_messages = True, input_key = "input", max_token_limit=3000)
st.title("Personalized Lesson")
st.write("Lesson Configuration")
with st.expander("Create and configure the lesson"):
    st.info("Use this section to upload your pdf file and give a module name.")
    # st.subheader("Lesson Configuration")
    # User inputs for configuration
    uploaded_file = st.file_uploader("Choose a PDF file for lesson content", type="pdf")
    module_name = st.text_input("Module/lesson name", "Export")
    # tool_name = st.text_input("Tool name (backend stuff)", "Intermediate_Answer", disabled=True)
    # tool_desc = st.text_area("Tool description (backend stuff)", f"Search for information about any '{module_name}' content. For any questions about '{module_name}', this tool should be used!", disabled=True)
    model_name = st.selectbox(
        "Pick a model",
        ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4"],
        index=2  # Default to "gpt-4o-mini"
    )
    model_temp = st.slider("Model temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.0)

    # Display the assistant response
    if st.button('Create this lesson'):
        index_name = format_index_name(module_name=module_name)
        if checkIfNewIndex(index_name):
            st.session_state['index_name'] = index_name
            st.session_state.update()
            if uploaded_file:
                # print("Uploaded file:", uploaded_file)
                # Extract text from the PDF
                all_text = extract_text_from_pdf(uploaded_file)
                # Generate summaries and topics
                with st.spinner('Summarizing text...'):
                    sums, topics = summarize_text(long_text=all_text, model=model_name, temp=model_temp)
                
                # In actual, you have to save this in the db
                st.session_state['topics'] = topics
                st.session_state.update()
                with st.spinner('Creating new index...'):
                    createNewVectorIndex(summaries=sums,index_name=st.session_state['index_name'])
                st.info("Lesson was created successfully.")
            else:
                st.error("Please upload a pdf file.")
        else:
            st.session_state['index_name'] = index_name
            st.session_state.update()
            st.info("The module/lesson with this name already exists. We will use the existing one.")

# Handle user queries
st.write("Enhance Personalization")
with st.expander("Modify user information"):
    st.info("Use this section to add user's information that could affect the personalization experience.")
    client_personal_info = st.text_area("User Personal Information", """Name: Jane Doe
    Age: 34
    Education: MBA in Marketing
    Experience: 10 years in small business management
    Email: jane.doe@example.com
    Phone: +1-234-567-8901
    Location: Springfield, IL""")
    client_business_info = st.text_area("User Business Information", """Business Name: Doe's Delights
    Industry: Food and Beverage
    Business Type: Small bakery specializing in organic and gluten-free products
    Year Established: 2015
    Number of Employees: 15
    Annual Revenue: $500,000
    Location: 123 Main Street, Springfield, IL

    Business Summary:
    Doe's Delights is a local bakery that has carved a niche in the organic and gluten-free segment. Founded in 2015 by Jane Doe, the business has steadily grown, catering to health-conscious consumers in Springfield and surrounding areas. The bakery offers a variety of baked goods, including bread, cakes, cookies, and pastries, all made with high-quality, organic ingredients.

    Goals:
    - Increase annual revenue by 20% over the next year
    - Expand the product line to include vegan options
    - Open a second location in the neighboring town
    - Enhance online presence and e-commerce capabilities

    Difficulties:
    - Managing supply chain disruptions for organic ingredients
    - Competition from larger bakeries and supermarkets
    - Limited marketing budget
    - Challenges in scaling up production without compromising quality""")
    language = st.text_input("User Preffered Language", "English")
    st.write("Note: In reality, this information will be fetched from user's profile and will be used when creating lessons.")

st.markdown("---")
st.write("Start Learning Here 👇")

if st.button("Click this to start the lesson"):
    st.session_state['started'] = True
    user_input = f"Teach me {module_name}"
    
    # Fetch vectorstore and initialize assistant
    with st.spinner('Getting the index from Pinecone...'):
        vectorstore = getExistingVectorStore(index_name=st.session_state['index_name'])
        if vectorstore is None:
            st.error("Create the lesson first.")
        else:
            with st.spinner('Loading learning assistant...'):
                st.session_state['assistant'] = createLearningAgent(
                    vectorstore=vectorstore,
                    tool_name="Intermediate_Answer",
                    tool_desc="Always use this for lesson.",
                    model_name=model_name,
                    model_temp=model_temp,
                    topics=st.session_state.get('topics', []),
                    language=language
                )
    
            if st.session_state['assistant'] is None:
                st.error("Failed to create learning agent.")
                st.stop()
    
            response = st.session_state['assistant'].invoke({
                "input": user_input,
                'client_personal_info': client_personal_info,
                'client_business_info': client_business_info
            })
            # st.write("Assistant:", response['output'])
            st.session_state['doc_messages'].append({"role": "assistant", "content": response['output']})

st.markdown("---")
st.subheader("The conversation panel")
# Display previous chat messages
for message in st.session_state['doc_messages']:
    with st.chat_message(message['role']):
        st.write(message['content'])

if user_query := st.chat_input("Enter your query here"):
    # Append user message
    st.session_state['doc_messages'].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if 'assistant' in st.session_state and st.session_state['assistant'] is not None:
        with st.spinner():
            response = st.session_state['assistant'].invoke({
                "input": user_query,
                'client_personal_info': client_personal_info,
                'client_business_info': client_business_info
            })
    else:
        response['output'] = "Learning assistant not initialized. Please create the lesson first and click start the lesson button."

    st.session_state['doc_messages'].append({"role": "assistant", "content": response['output']})
    with st.chat_message("assistant"):
        st.markdown(response['output'])