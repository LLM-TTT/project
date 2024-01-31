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
            
        print(patent_data)

        for cluster in data["results"]["cluster"]:
            for result in cluster["result"]:
                id = result["patent"]["publication_number"]
                if id not in patent_data.keys():
                    patent_data[id] = {
                        "pdf": result["patent"]["pdf"],
                    }
            
            # for a in data['results']['cluster'][0]['result']:
            #     if not a['patent']['pdf']: #control if there is an existing url in the meta data
            #         print("No URL found in meta data for PDF #",count,"This PDF will be skipped.")
            #         continue
            #     loader = PyPDFLoader(patent_base_url+a['patent']['pdf'])
            #     pdf = loader.load_and_split()
            #     count+=1
            #     if pdf == []: #Falls PDF nicht maschinenlesbar ist, dieses überspringen; vllt. noch extra Liste anlegen mit atypischen PDFs
            #         print("PDF nicht Maschinenlesbar")
            #     else:
            #         pdf_list.append(pdf[0])
            #         print("PDF #",count,"erfolgreich zur Liste hinzugefügt.")

        #Loop for multiple Google Patents API calls with Classifications     
        for id in patent_data.keys():
        
            # generating Google Patent links for each ID

            url = "https://patents.google.com/patent/" + id + "/en"

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
                
            patent_data[id].update({
                "title": title,
                "abstract": abstract,
                "description": description,
                "claims": claims
            })
        
        print(patent_data)
        # for i in class_list:
        #     openai_response = i #Search String for Google Patents
        #     url_base = "https://serpapi.com/search.html?engine=google_patents"
        #     query = openai_response.replace(" ", "+")
        #     url = url_base + "&q=" + query + "&api_key=" + patent_api_key

        #     # API call Google Patents
        #     response = requests.get(url)

        #     # Check if API call was successful
        #     if response.status_code == 200:
        #         data = response.json() #write json-answer in var
        #         print("API Call für '",openai_response,"' erfolgreich:",data)
        #     else:
        #         print(f"Error with API request: Status code {response.status_code}")

        #     for a in data['results']['cluster'][0]['result']:
        #         print("Lese Link ein:",patent_base_url+a['patent']['pdf'])
        #         if not a['patent']['pdf']: #control if there is an existing url in the meta data
        #             print("No URL found in meta data for PDF #",count,"This PDF will be skipped.")
        #             continue
        #         loader = PyPDFLoader(patent_base_url+a['patent']['pdf'])
        #         pdf = loader.load_and_split()
        #         count+=1
        #         if pdf == []: #Falls PDF nicht maschinenlesbar ist, dieses überspringen; vllt. noch extra Liste anlegen mit atypischen PDFs
        #             print("PDF nicht Maschinenlesbar")
        #         else:
        #             pdf_list.append(pdf[0])
        #             print("PDF #",count,"erfolgreich zur Liste hinzugefügt.")
        progress(0.6, desc="Collecting patents")

        # extracting patent ids + abstracts for further prompt usage

        abstract_prompt = ""

        for patent_id, patent_info in patent_data.items():
            # Check if there is an abstract for the patent
            if patent_info['abstract']:
                abstract_prompt = abstract_prompt + f'{patent_id}: "{patent_info["abstract"]}"\n'

        for keyword in keywords_list:
            url_base = "https://serpapi.com/search.html?engine=google_patents"
            query = keyword.replace(" ", "+")
            url = url_base + "&q=" + query + "&api_key=" + patent_api_key

            print(url)
                
        #Login MongoDB with User and specific database
        uri = "mongodb+srv://timmey:faB8MFdyyb7zWvVr@llm-ttt.8kqrnka.mongodb.net/?retryWrites=true&w=majority"

        # Create a new client and connect to the server
        client = MongoClient(uri, server_api=ServerApi('1'))

        DB_NAME = "llm-ttt"
        COLLECTION_NAME = "pdfresults"
        ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

        MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

        # insert the documents in MongoDB Atlas with their embedding
        vector_search = MongoDBAtlasVectorSearch.from_documents( # !!--> AttributeError: 'str' object has no attribute 'page_content'!!
            documents=patent_data, 
            embedding=OpenAIEmbeddings(disallowed_special=()),
            collection=MONGODB_COLLECTION,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        )
        # Perform a similarity search with Score between the embedding of the query and the embeddings of the documents
        progress(0.9, desc="Compare the patents")
        query = str(content)

        results = vector_search.similarity_search_with_score(
            query=query,
            k=5, #Output for the top n results
        )

        # Display results
        for result in results:
            print(result)

        formatted_results = []
        formatted_result = ""
        for result in results:
            formatted_result = ("Titel: {}; Übereinstimmung: {}%; Quelle: {}".format(result[0].metadata['title'] ,round(result[1] * 100, 2), result[0].metadata['source']))
        # Append the formatted result to the list
            formatted_results.append(formatted_result)
        
        #result.live(formatted_result)
        return formatted_results


file_path = "../data_dump"
def create_pdf(file_path, formatted_results):

    pdf_file = canvas.Canvas(file_path)

    # Set font and size
    pdf_file.setFont("Helvetica", 12)

    y_coordinate = 700

    
    # Draw each line of formatted result
    pdf_file.drawString(100, y_coordinate, formatted_results)
    # Move down the y-coordinate for the next line
    y_coordinate -= 20  

    # Save the PDF
    pdf_file.save()

    return pdf_file

# image_path = 'https://drive.google.com/file/d/1wqrLEadHAt7xl4djVx4lHu7ts_8KOxme/view?usp=sharing'
# absolute_path = os.path.abspath(image_path)

def clear_db(): #clear the vector database
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
            result = gr.Textbox(label="Your Abstract", interactive=False)      
            
            keywords = gr.Textbox(label="Key Words", value="None", interactive=False) #New Value "<List of Key Words>"
            classes = gr.Textbox(label="Classifications", value="None", interactive=False) #New Value "<List of Classifications>"

            result.change(output_keywords, [result, slide_keywords], keywords) 
            result.change(output_classes, [result, slide_classes], classes) 

            endresult = gr.Textbox(label="End Result", value="None") #New Value "Top 5 PDFs ...."
            
            classes.change(patent_analysis_rest, [result, keywords, classes], endresult) #It does not matter if you choose classes or keywords from above

            # with gr.Accordion(label= "Technical Details", open=False):   

            #     if output_classes == 1:
            #         api_openai = gr.HighlightedText([("Current State of API: ", None), ("Connection", "Successfull")], color_map={"Successfull": "green", "Failed": "red"})
            #     else:
            #         api_openai = gr.HighlightedText([("Current State of API: ", None), ("Connection", "Failed")], color_map={"Successfull": "green", "Failed": "red"})
            #     gr.Textbox(label="API OpenAI", value="Disconnected") #New Value "Connected"
            #     gr.Textbox(label="API Patent Database #1", value="Disconnected") #New Value "Connected"
            #     gr.Textbox(label="API Connection Vector Database", value="Disconnected") #New Value "Connected"
            #     gr.Textbox(label="API Calls", value="Disconnected") #Anzahl API Calls live aktualisieren die duirchgeführt wurden 
            #     gr.Textbox(label="PDF List", value="No PDFs added yet") #Anzahl PDFs die hinzugefügt wurden; New Value "xx PDFs added to the list."
            #     vector_db = gr.Textbox(label="Collection Vector Database", value="No PDFs added yet") #New Value "xx PDFs added to the collection of vector database."
            pdf_file = gr.Button("Create PDF")
            clear_button = gr.Button("New Research")
            clear_button.click(clear_db,outputs=[endresult])    
            with gr.Row():
                    outputs = "file"
                    pdf_file.click(create_pdf, inputs=[endresult], outputs=[pdf_file])
            
            button.click(patent_analysis, inputs=[files], outputs=[result])

                   
demo.launch()

# <a href="https://de.freepik.com/fotos-kostenlos/geschaeftsmann-haelt-gelbe-gluehbirne-mit-kopierraum-fuer-geschaeftsloesung-und-kreatives-denken-ideenkonzept-durch-3d-darstellung_26791662.htm#query=patente&position=5&from_view=search&track=sph&uuid=7931811d-4408-4f21-a68a-3450f7e46c8d">Bild von DilokaStudio</a> auf Freepik
