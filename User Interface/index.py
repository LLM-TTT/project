#!C:\Users\ruhmt\AppData\Local\Programs\Python\Python311\python.exe
import streamlit as st
import PyPDF2
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from Levenshtein import distance
import datetime
import os
import openai
import requests
import json
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from pymongo.server_api import ServerApi

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
    info_screen = st.empty()
    info_screen.info('Wait a minute. Your patent will be analyzed!', icon="👨‍💻")
    # Read the PDF file
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    # Extract the content
    content = ""
    for page in range(len(pdf_reader.pages)):
        content += pdf_reader.pages[page].extract_text()
    # Display the content
    st.write(content)

    prompt1 = f"""
    The following abstract descripes a concept for a novel invention:\
    ```{content}```\
    Name 5 key words based on this abstract, that I can use for the search in a patent database. \
    Optimize the key words to get back more results. Result as python string.
    """

    prompt2 = f"""
    The following abstract descripes a concept for a novel invention:\
    ```{content}```\
    Name 5 CPC classifications based on this abstract, that I can use for the search in a patent database. \
    Please give me a python string for the codes of the 5 most relevant \
    CPC classifications to a possible patent. 
    """

    response_keywords = get_completion(prompt1)
    response_classes = get_completion(prompt2)

    #cast the results (key words) from string to list
    keywords_list = []

    splitstring = response_keywords.split(", ") #split the key words
    for i in splitstring:
        keywords_list.append(i[1:-1]) #remove the quotation marks
    print(keywords_list)

    #cast the results (classifications) from string to list
    class_list = []

    new_string = response_classes.replace(",", "",9999) #remove commas
    splitstring = new_string.split() #split the classes
    for i in splitstring:
        class_list.append(i[1:-1]) #remove the quotation marks
    print(class_list)

    #initialization of base vars for the following loop
    patent_api_key = os.environ['GOOGLE_PATENT_API_KEY']
    pdf_list = []
    count = 0
    patent_base_url = "https://patentimages.storage.googleapis.com/" #just to complete the url

    short_keywords_list = keywords_list[3:]
    short_class_list = class_list[3:]

    #Loop for multiple Google Patents API calls with Key Words
    for i in short_keywords_list:
        openai_response = i #Search String for Google Patents
        url_base = "https://serpapi.com/search.html?engine=google_patents"
        query = openai_response.replace(" ", "+")
        url = url_base + "&q=" + query + "&api_key=" + patent_api_key

        # API call Google Patents
        response = requests.get(url)

        # Check if API call was successful
        if response.status_code == 200:
            data = response.json() #write json-answer in var
            print("API Call für '",openai_response,"' erfolgreich:",data)
        else:
            print(f"Error with API request: Status code {response.status_code}")

        for a in data['results']['cluster'][0]['result']:
            if not a['patent']['pdf']: #control if there is an existing url in the meta data
                print("No URL found in meta data for PDF #",count,"This PDF will be skipped.")
                continue
            loader = PyPDFLoader(patent_base_url+a['patent']['pdf'])
            pdf = loader.load_and_split()
            count+=1
            if pdf == []: #Falls PDF nicht maschinenlesbar ist, dieses überspringen; vllt. noch extra Liste anlegen mit atypischen PDFs
                print("PDF nicht Maschinenlesbar")
            else:
                pdf_list.append(pdf[0])
                print("PDF #",count,"erfolgreich zur Liste hinzugefügt.")
    #Loop for multiple Google Patents API calls with Classifications
    for i in short_class_list:
        openai_response = i #Search String for Google Patents
        url_base = "https://serpapi.com/search.html?engine=google_patents"
        query = openai_response.replace(" ", "+")
        url = url_base + "&q=" + query + "&api_key=" + patent_api_key

        # API call Google Patents
        response = requests.get(url)

        # Check if API call was successful
        if response.status_code == 200:
            data = response.json() #write json-answer in var
            print("API Call für '",openai_response,"' erfolgreich:",data)
        else:
            print(f"Error with API request: Status code {response.status_code}")

        for a in data['results']['cluster'][0]['result']:
            print("Lese Link ein:",patent_base_url+a['patent']['pdf'])
            if not a['patent']['pdf']: #control if there is an existing url in the meta data
                print("No URL found in meta data for PDF #",count,"This PDF will be skipped.")
                continue
            loader = PyPDFLoader(patent_base_url+a['patent']['pdf'])
            pdf = loader.load_and_split()
            count+=1
            if pdf == []: #Falls PDF nicht maschinenlesbar ist, dieses überspringen; vllt. noch extra Liste anlegen mit atypischen PDFs
                print("PDF nicht Maschinenlesbar")
            else:
                pdf_list.append(pdf[0])
                print("PDF #",count,"erfolgreich zur Liste hinzugefügt.")

    #Login MongoDB with User and specific database
    uri = "mongodb+srv://timmey:faB8MFdyyb7zWvVr@llm-ttt.8kqrnka.mongodb.net/?retryWrites=true&w=majority"

    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))

    DB_NAME = "llm-ttt"
    COLLECTION_NAME = "pdfresults"
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

    MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

    # insert the documents in MongoDB Atlas with their embedding
    vector_search = MongoDBAtlasVectorSearch.from_documents(
        documents=pdf_list, 
        embedding=OpenAIEmbeddings(disallowed_special=()),
        collection=MONGODB_COLLECTION,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )
    # Perform a similarity search with Score between the embedding of the query and the embeddings of the documents
    query = str(content)

    results = vector_search.similarity_search_with_score(
        query=query,
        k=5,
    )

    # Display results
    for result in results:
        print(result)

    for result in results:
        st.write("Übereinstimmung:",round(result[1]*100,2),"%; Quelle:", result[0].metadata['source'])

    # st.write(response_keywords)
    # st.write("----")
    # st.write(response_classes)
    info_screen.empty()

    # #if response_classes is not None:
    # #    #openai_response sient nur als Beispiel; wir müssen noch alle Keywords einzeln extrahieren
    # #    openai_response = 'Secure Access to Vehicles using Biometric'
    # #    url_base = "https://serpapi.com/search.html?engine=google_patents"
    # #    query = openai_response.replace(" ", "+")
    # #    url = url_base + "&q=" + query + "&api_key=" + patent_api_key

    #     # API call
    #     response = requests.get(url)

    #     # Check if API call was successful
    #     if response.status_code == 200:
    #         # extract JSON data from answer
    #         data = response.json()

    #         # save JSON data in a file
    #         filename = "../data_dump/" + query + ".json"
    #         with open(filename, 'w') as file: #FileNotFoundError: [Errno 2] No such file or directory: '../data_dump/Secure+Access+to+Vehicles+using+Biometric.json'
    #             json.dump(data, file, indent=4)

    #         f = open('../data_dump/'+query+'.json') #Load Master List

    #         data = json.load(f)

    #         counter = 0
    #         for i in data:
    #             counter += 1
    #             st.write("#"+str(counter),"Titel:",i['title'])
    #             st.write("PDF: "+i['pdf'])
    #     else:
    #         print(f"Error with API request: Status code {response.status_code}")