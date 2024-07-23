from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import json
from openai import OpenAI
from prompts import PromptsCollection
import PyPDF2
from io import BytesIO
from agent_utils import summarize_topics

def extract_text_from_pdf(file_like_object):
    # Ensure the file-like object is in binary read mode
    with BytesIO(file_like_object.read()) as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() or ""
    return text

# def extract_text_from_pdf(pdf_path:str = None):
#     with open(pdf_path, 'rb') as file:
#         reader = PyPDF2.PdfReader(file)
#         text = ""
#         for page_num in range(len(reader.pages)):
#             page = reader.pages[page_num]
#             text += page.extract_text()
#     return text

def summarize_text(long_text:str = None, chunk_size:int = 2000, chunk_overlap:int = 20, model="gpt-3.5-turbo", temp=0):
    # Split text into chunks using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(long_text)
    summaries = []
    topics = ''

    # Organize summaries
    for chunk in chunks:
        result = summarize_topics(chunk=chunk, model=model, temp=temp)
        # print("Result:", result)
        summaries.append(result['summary'])
        topics += result['topics']
        # print(type(result))
        # json_result = json.loads(result)

        # summaries.append(json_result['summary'])
        # topics += json_result['topics']

    return (summaries, topics)

def format_index_name(module_name: str) -> str:
    # Lowercase the module name and replace spaces with hyphens
    formatted_name = module_name.lower().replace(" ", "-")
    index_name = f"{formatted_name}-lesson-index"
    return index_name

# Summarize each chunk using OpenAI API function call (not good that much)
# def summarize_topics(chunk):

#     client = OpenAI()
#     prompt = PromptsCollection.summary_prompt.format(chunk=chunk)
#     completion = client.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return completion.choices[0].message.content