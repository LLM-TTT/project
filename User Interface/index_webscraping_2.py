import gradio as gr
import PyPDF2
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from Levenshtein import distance
import datetime
import time
import os
from bs4 import BeautifulSoup as bs
import openai
import requests
import json
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from reportlab.pdfgen import canvas
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# close all open ports
gr.close_all()

#Load OpenAI API Key and Google Patent API Key
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']
patent_api_key = os.environ['GOOGLE_PATENT_API_KEY']

#Define LLM Model
llm_model = "gpt-4"

#Initialize LLM Model Attributes
from openai import OpenAI

client = OpenAI()
def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}] #role: define the role of the llm; conent: how the llm should act
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, #creativity range 0..1
    )
    return response.choices[0].message.content

#Load PDF File from User Input and extract first Page
def patent_analysis(file, progress=gr.Progress()):
    if file is not None:
        #Read the PDF file
        progress(0.2, desc="Reading the file")
        pdf_reader = PyPDF2.PdfReader(file)
        # Extract the content
        content = ""
        for page in range(len(pdf_reader.pages)):
            content += pdf_reader.pages[page].extract_text()
        # Analyzing the File  
        progress(0.3, desc="Analyzing the file")
    return content

#LLM Prompt generating Key Words on base of the PDF Input from User
def output_keywords(content, n, progress=gr.Progress()):
    progress(0, desc="Generating Key Words...")
    prompt1 = f"""
    The following abstract descripes a concept for a novel invention:\
    ```{content}```\
    Name {n} key words based on this abstract, that I can use for the search in a patent database. \
    Optimize the key words to get back more results. Result as python string.
    """

    response_keywords = get_completion(prompt1)

    return response_keywords

#LLM Prompt generating Classifications on base of the PDF Input from User
def output_classes(content, n, progress=gr.Progress()):
    progress(0, desc="Generating Classifications...")
    prompt2 = f"""
        The following abstract descripes a concept for a novel invention:\
        ```{content}```\
        Name {n} CPC classifications based on this abstract, that I can use for the search in a patent database. \
        Please give me a python string for the codes of the {n} most relevant \
        CPC classifications to a possible patent. 
        """
    
    response_classes = get_completion(prompt2)
    
    return response_classes

#Clearing vector database
def clear_db():
    uri = "mongodb+srv://timmey:faB8MFdyyb7zWvVr@llm-ttt.8kqrnka.mongodb.net/?retryWrites=true&w=majority"

    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))

    DB_NAME = "llm-ttt"
    COLLECTION_NAME = "pdfresults"
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

    MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]
    
    x = MONGODB_COLLECTION.delete_many({})
    delete = str(x.deleted_count) + " documents deleted."
    return delete

#Continue with API Calls, vectorizing and vector database handling
def patent_analysis_rest(content, response_keywords, response_classes, progress=gr.Progress()):
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
        progress(0.5, desc="Research for patents")
        patent_api_key = os.environ['GOOGLE_PATENT_API_KEY']
        count = 0
        patent_base_url = "https://patentimages.storage.googleapis.com/" #just to complete the url
        patent_data = {}

        #Loop for multiple Google Patents API calls with Key Words
        for i in keywords_list:
            openai_response = i #Search String for Google Patents
            url_base = "https://serpapi.com/search.html?engine=google_patents"
            query = openai_response.replace(" ", "+")
            url = url_base + "&q=" + query + "&api_key=" + patent_api_key

            # API call Google Patents
            response = requests.get(url)

            # Check if API call was successful
            if response.status_code == 200:
                data = response.json() #write json-answer in var
                for cluster in data["results"]["cluster"]:
                    for result in cluster["result"]:
                        id = result["patent"]["publication_number"]
                        if id not in patent_data.keys():
                            patent_data[id] = {
                                "pdf": result["patent"]["pdf"],
                            }
            else:
                print(f"Error with API request: Status code {response.status_code}")

        for cluster in data["results"]["cluster"]:
            for result in cluster["result"]:
                patent_id = result["patent"]["publication_number"]
                if patent_id not in patent_data.keys():
                    patent_data[patent_id] = {
                        "pdf": result["patent"]["pdf"],
                    }

        #Scraping complete patent data
        progress(0.6, desc="Collecting patent data")

        for patent_id in patent_data.keys():
        
            # generating Google Patent links for each ID

            url = "https://patents.google.com/patent/" + patent_id + "/en"

            response = requests.get(url)
            html_content = response.content
            soup = bs(html_content, 'html.parser')

            # Scraping Title

            title_span = soup.find('span', itemprop='title')

            if title_span is not None:
                title = title_span.get_text()

                # Removing weird ending of title
                to_remove = "\n"
                title = title.replace(to_remove, "").strip()
            else:
                title = False

            # Scraping Abstract

            abstract_div = soup.find('div', class_='abstract')

            if abstract_div is not None:
                abstract = abstract_div.get_text()
            else:
                abstract = False 

            # Scraping Description

            description_section = soup.find('section', itemprop='description')

            if description_section:

                # Removing H2 from section
                h2_tag = description_section.find('h2')
                if h2_tag:
                    h2_tag.decompose()
                    
                # Removing all 'notranslate' class items
                for notranslate_tag in description_section.find_all(class_='notranslate'):
                    notranslate_tag.decompose()
                    
                # Removing all <aside> elements
                for aside_tag in description_section.find_all('aside'):
                    aside_tag.decompose()

                # Extracting and joining the text
                description = "".join(description_section.stripped_strings)
                if description == "":
                    description = False

            else:
                description = False   

            # Scraping Claims

            description_section = soup.find('section', itemprop='claims')

            if description_section:
                # Removing H2 from section
                h2_tag = description_section.find('h2')
                if h2_tag:
                    h2_tag.decompose()
                    
                # Removing all 'notranslate' class items
                for notranslate_tag in description_section.find_all(class_='notranslate'):
                    notranslate_tag.decompose()
                    
                # Removing all <aside> elements
                for aside_tag in description_section.find_all('aside'):
                    aside_tag.decompose()

                # Extracting and joining the text
                claims = "".join(description_section.stripped_strings)
                if claims == "":
                    claims = False

            else:
                claims = False
                
            patent_data[patent_id].update({
                "title": title,
                "abstract": abstract,
                "description": description,
                "claims": claims
            })

        # Converting the patent data into a usable format to perform a vector search
        patent_list = []

        for patent_id, data in patent_data.items():
            page_content = f"{data['title']} {data['abstract']} {data['description']} {data['claim']}"
            metadata = {"patent_id": patent_id}
            patent_list.append(Document(page_content=page_content, metadata=metadata))


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
            documents=patent_list,
            embedding=OpenAIEmbeddings(disallowed_special=()),
            collection=MONGODB_COLLECTION,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        )

        #pause for the db to save
        time.sleep(5)

        # Perform a similarity search with Score between the embedding of the query and the embeddings of the documents
        progress(0.9, desc="Compare the patents")
        query = str(content)

        results = vector_search.similarity_search_with_score(
            query=query,
            k=20, #Output for the top n results
        )

        vector_result = {}

        for result in results:
            vector_result[result[0].metadata['patent_id']] = result[1]
        
        comparison_prompt = f"""The following texts are abstracts from patent specifications. Your task is to compare the "Testing Abstract" to all the others. 
        It is important that you focus on comparing the concepts that the abstracts describe, not the way they are written. 
        Rank the remaining abstracts on how well they match with the Testing Abstract by giving them a rating from 0 to 10 points. 
        0 meaning they have absolutely nothing in common and 10 meaning they basically describe the exact same idea.
        Your output should be a python dictionary with the title "comparison", each element hast the Abstract number as key and the rating as value.
        I want to convert your output string to an actual dictionary, so make sure the formatting is right.

        Testing Abstract: "{content}"
        """

        for patent_id in vector_result.keys():
            # Check if there is an abstract for the patent
            if patent_id in patent_data.keys():
                comparison_prompt = comparison_prompt + f'{patent_id}: "{patent_data[patent_id]["abstract"]}"\n'

        final_result = get_completion(comparison_prompt)

        return final_result


with gr.Blocks(theme=gr.themes.Glass(primary_hue=gr.themes.colors.zinc, secondary_hue=gr.themes.colors.gray, neutral_hue=gr.themes.colors.gray)) as demo:
    gr.Markdown("# Patent Pete")
    gr.Markdown("<p style='font-size:16px;'>Hi, I'm Pete your assistance for patent researchs. I help you with finding out, if your patent already exists. Let's start with uploading your idea!</p>")
#Alternative Mercedes Benz Theme: gr.themes.Glass(primary_hue=gr.themes.colors.zinc, secondary_hue=gr.themes.colors.gray, neutral_hue=gr.themes.colors.gray)
    with gr.Row():

        with gr.Column() as main:
            
            gr.Markdown("<p><h1>Input</h1></p>")

            files= gr.File(file_types=['.pdf'], label="Upload your pdf here.")

            gr.Markdown("<u>Configuration Options</u>")

            gr.Radio(["International Patent Classification (IPC)", "United States Patent and Trademark Office (USPTO)", "Cooperative Patent Classification (CPC)", "Deutsche Klassifizierung (DEKLA)"], label="Type of Classification", value="Cooperative Patent Classification (CPC)"),

            slide_keywords = gr.Slider(1, 5, step=1, value=2, label="Number of Key Words")
            slide_classes = gr.Slider(1, 5, step=1, value=2, label="Number of Classifications")

            gr.CheckboxGroup(["Google Patents", "Espacenet", "European Patent Office (EPO)", "DEPATISnet"], label="Databases", info="Which databases should be searched?", value="Google Patents"),

            button = gr.Button("Submit")
            

        with gr.Column() as sidebar_right:
            gr.Markdown("<p><h1>Output</h1></p>")
            result_output = gr.Textbox(label="Your Abstract", interactive=False)      
            
            keywords = gr.Textbox(label="Key Words", value="None", interactive=False) #New Value "<List of Key Words>"
            classes = gr.Textbox(label="Classifications", value="None", interactive=False) #New Value "<List of Classifications>"

            result_output.change(output_keywords, [result_output, slide_keywords], keywords) 
            result_output.change(output_classes, [result_output, slide_classes], classes) 

            endresult = gr.Textbox(label="End Result", value="None") #New Value "Top 5 PDFs ...."
            
            classes.change(patent_analysis_rest, [result_output, keywords, classes], endresult) #It does not matter if you choose classes or keywords from above

            clear_button = gr.Button("New Research")
            clear_button.click(clear_db,outputs=[endresult])    
            
            button.click(patent_analysis, inputs=[files], outputs=[result_output])

                   
demo.launch()