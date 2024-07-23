from utils import (
    extract_text_from_pdf,
    summarize_text,
    format_index_name
)
from agent_utils import createLearningAgent, createNewVectorIndex

from dotenv import load_dotenv
import os
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["PINECONE_API_KEY"] = pinecone_api_key


# Extract text from the PDF
pdf_path = './data/export.pdf'
all_text = extract_text_from_pdf(pdf_path)
# Generate summaries and topics
sums, topics = summarize_text(long_text=all_text)

# Define tool name and desc (Ask from User)
module_name = "Export"
tool_name = "Intermediate_Answer"
tool_desc = "Search for information about any Export learning content. For any questions about learning content, you must use this tool!"
language = "English"
client_personal_info = """
Name: Jane Doe
Age: 34
Education: MBA in Marketing
Experience: 10 years in small business management
Email: jane.doe@example.com
Phone: +1-234-567-8901
Location: Springfield, IL
"""
client_business_info = """
Business Name: Doe's Delights
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
- Challenges in scaling up production without compromising quality
"""
model_name = "gpt-4o-mini"
model_temp = 0
index_name = format_index_name(module_name=module_name)

# Creator Exp
createNewVectorIndex(summaries=sums, index_name=index_name)

# User Exp
assistant = createLearningAgent(
    index_name=index_name,
    tool_name=tool_name,
    tool_desc=tool_desc,
    model_name=model_name,
    model_temp=model_temp,
    topics=topics,
    language=language
)

# When user clicks start button, run this
user_input = f"Teach me {module_name}"
response = assistant.invoke({"input": user_input, 'client_personal_info':client_personal_info, 'client_business_info':client_business_info})
print("Assistant:", response['output'])

# After that, get user query and run the following
while True:
    user_input = input("You:")
    response = assistant.invoke({"input": user_input, 'client_personal_info':client_personal_info, 'client_business_info':client_business_info})
    print("AI:", response['output'])