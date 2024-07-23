from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.agents import AgentExecutor, create_openai_tools_agent
from prompts import PromptsCollection
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from output_parser import LessonContent
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["PINECONE_API_KEY"] = pinecone_api_key

pc = Pinecone()
embedding_model = OpenAIEmbeddings()

def checkIfNewIndex(index_name:str) -> bool:
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name in existing_indexes:
        return False
    else:
        return True
    
def createNewVectorIndex(summaries:list = None, index_name:str = None) -> None:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
    # Create vectorstore
    vectorstore = PineconeVectorStore.from_texts(
        summaries,
        index_name=index_name,
        embedding=embedding_model
    )


def getExistingVectorStore(index_name):
    try:
        vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding_model)
    except:
        return None
    return vectorstore

def getRetriever(vectorstore = None):

    retriever = vectorstore.as_retriever(search_type="mmr")
    return retriever

def createTool(retriever = None, tool_name = "Intermediate_Answer", tool_desc = "Use this tool for every task."):
    tool = create_retriever_tool(
        retriever,  # The RAG chain instance you are using
        tool_name,  # Renamed to better reflect the focus on learning materials
        tool_desc
    )
    return tool

def createLearningAgent(vectorstore, tool_name, tool_desc, model_name, model_temp, topics, language, memory=None):
    #To retriever
    retriever = vectorstore.as_retriever(search_type="mmr")
    # To tool
    tool = create_retriever_tool(
        retriever,  # The RAG chain instance you are using
        tool_name,  # Renamed to better reflect the focus on learning materials
        tool_desc
    )
    # Initialize tools
    tools = [tool]
    # Define llm
    llm = ChatOpenAI(model=model_name, temperature=model_temp)
    # To Agent
    custom_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PromptsCollection.reserved_prompt),
            MessagesPlaceholder("chat_history", optional=False),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    custom_prompt = custom_prompt.partial(topics_to_cover=topics, language=language)
    

    # Create an agent with all tools
    learning_agent = create_openai_tools_agent(llm, tools, custom_prompt)

    if memory is None:
        memory = ConversationBufferWindowMemory(memory_key = "chat_history", return_messages = True, input_key = "input", max_token_limit=3000)
    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent = learning_agent, tools = tools, verbose = False, memory = memory, handle_parsing_errors=True)

    return agent_executor


# to generate summary and topics
def summarize_topics(chunk:str =None, model:str ='gpt-3.5-turbo', temp:float =0) -> dict[str, str]:
    '''
    To summarize chunk and generate topics from that summary
    '''
    llm = ChatOpenAI(model=model, temperature=temp)
    output_parser = JsonOutputParser(pydantic_object=LessonContent)

    summary_prompt = PromptTemplate(
        # template="Answer the user query.\n{format_instructions}\n{query}\n",
        template = PromptsCollection.summary_prompt,
        input_variables=["chunk"],
        partial_variables={"format_instruction": output_parser.get_format_instructions()},
    )

    chain = summary_prompt | llm | output_parser

    try:
        output = chain.invoke({"chunk":chunk})
    except:
        output = {'summary':'','topics':''}
    return output