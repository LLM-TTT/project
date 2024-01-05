#!C:\Users\ruhmt\AppData\Local\Programs\Python\Python311\python.exe
import streamlit as st
import PyPDF2
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
import datetime
import os
import openai
import requests
import json

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']
patent_api_key = os.environ['GOOGLE_PATENT_API_KEY']

llm_model = "gpt-4"


def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

print("Content-type: text/html")
print()



with st.sidebar:
    st.title('LLM-TTTM')


#with st.form(key='search_form'):
#    st.text_input("Search patents", key="phrases")
#    st.form_submit_button(label="Search") 

import streamlit as st

st.title("Patent Pete")

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
if uploaded_file is not None:
    # Read the PDF file
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    # Extract the content
    content = ""
    for page in range(len(pdf_reader.pages)):
        content += pdf_reader.pages[page].extract_text()
    # Display the content
    st.write(content)

    prompt1 = f"""```{content}```\
    The abstract above describes a concept for a novel invention.\
    I would like to search a patent database to find out whether \
    there are already patents for such a concept. Name 5 phrases that I can \
    use for the search. Each phrase should contain between 5 to 10 words. \
    Optimize the phrases to get back more results.
    """

    prompt2 = f"""```{content}```\
    The abstract above describes a concept for a novel invention.\
    I would like to search a patent database to find out whether \
    there are already patents for such a concept. Please list me the codes of the 5 most relevant \
    USPTO classifications to a possible patent for this concept without explanations for the codes.
    """

    response_keywords = get_completion(prompt1)
    response_classes = get_completion(prompt2)

    st.write(response_keywords)
    st.write("----")
    st.write(response_classes)

    if response_classes is not None:
        #openai_response sient nur als Beispiel; wir müssen noch alle Keywords einzeln extrahieren
        openai_response = 'Secure Access to Vehicles using Biometric'
        url_base = "https://serpapi.com/search.html?engine=google_patents"
        query = openai_response.replace(" ", "+")
        url = url_base + "&q=" + query + "&api_key=" + patent_api_key

        # API call
        response = requests.get(url)

        # Check if API call was successful
        if response.status_code == 200:
            # extract JSON data from answer
            data = response.json()

            # save JSON data in a file
            filename = "../data_dump/" + query + ".json"
            with open(filename, 'w') as file: #FileNotFoundError: [Errno 2] No such file or directory: '../data_dump/Secure+Access+to+Vehicles+using+Biometric.json'
                json.dump(data, file, indent=4)

            f = open('../data_dump/'+query+'.json') #Load Master List

            data = json.load(f)

            counter = 0
            for i in data:
                counter += 1
                st.write("#"+str(counter),"Titel:",i['title'])
                st.write("PDF: "+i['pdf'])
        else:
            print(f"Error with API request: Status code {response.status_code}")

        
