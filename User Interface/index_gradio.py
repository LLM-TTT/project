import gradio as gr
import PyPDF2
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from Levenshtein import distance
import datetime
import time
import os
import openai
import requests
import json
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from reportlab.pdfgen import canvas

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

def output_keywords(content, progress=gr.Progress()):
    progress(0, desc="Generating Key Words...")
    prompt1 = f"""
    The following abstract descripes a concept for a novel invention:\
    ```{content}```\
    Name 5 key words based on this abstract, that I can use for the search in a patent database. \
    Optimize the key words to get back more results. Result as python string.
    """

    response_keywords = get_completion(prompt1)

    return response_keywords

def output_classes(content, progress=gr.Progress()):
    progress(0, desc="Generating Key Words...")
    prompt2 = f"""
        The following abstract descripes a concept for a novel invention:\
        ```{content}```\
        Name 5 CPC classifications based on this abstract, that I can use for the search in a patent database. \
        Please give me a python string for the codes of the 5 most relevant \
        CPC classifications to a possible patent. 
        """
    
    response_classes = get_completion(prompt2)
    
    return response_classes



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
        pdf_list = []
        count = 0
        patent_base_url = "https://patentimages.storage.googleapis.com/" #just to complete the url

        short_keywords_list = keywords_list[4:]
        short_class_list = class_list[4:]

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
        progress(0.6, desc="Collecting patents")
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
        progress(0.9, desc="Compare the patents")
        query = str(content)

        results = vector_search.similarity_search_with_score(
            query=query,
            k=5,
        )

        # Display results
        for result in results:
            print(result)

        formatted_results = []
        for result in results:
            formatted_result = (
            "Übereinstimmung: {}%; Quelle: {}".format(
                round(result[1] * 100, 2),
                result[0].metadata['source']
            )
        )
        # Append the formatted result to the list
            formatted_results.append(formatted_result)
        
        #result.live(formatted_result)
        return formatted_result


file_path = "../data_dump"
def create_pdf(file_path, results):
    pdf = canvas.Canvas(file_path)

    # Set font and size
    pdf.setFont("Helvetica", 12)

    for result in results:
        pdf.drawString(100, 700, result[0].metadata['title'])
        pdf.drawString(100, 700, result[0].metadata['source'])

    # Save the PDF
    pdf.save()

    return pdf

# image_path = 'https://drive.google.com/file/d/1wqrLEadHAt7xl4djVx4lHu7ts_8KOxme/view?usp=sharing'
# absolute_path = os.path.abspath(image_path)

def test(data, data2, data3):
    new = "TEST: " + data + " UND " + data2 + " UUUUUUNNNNNNDDDD " + data3
    return new

with gr.Blocks(theme=gr.themes.Glass(primary_hue=gr.themes.colors.zinc, secondary_hue=gr.themes.colors.gray, neutral_hue=gr.themes.colors.gray)) as demo:
    gr.Markdown("# Patent Pete")
    gr.Markdown("<p style='font-size:16px;'>Hi, I'm Pete your assistance for patent researchs. I help you with finding out, if your patent already exists. Let's start with uploading your idea!</p>")
#Alternative Mercedes Benz Theme: gr.themes.Glass(primary_hue=gr.themes.colors.zinc, secondary_hue=gr.themes.colors.gray, neutral_hue=gr.themes.colors.gray)
    with gr.Row():

        with gr.Column() as main:
            
            gr.Markdown("<p><h1>Input</h1></p>")

            gr.Markdown("<u>Configuration Options</u>")

            gr.Radio(["International Patent Classification (IPC)", "United States Patent and Trademark Office (USPTO)", "Cooperative Patent Classification (CPC)", "Deutsche Klassifizierung (DEKLA)"], label="Type of Classification", value="Cooperative Patent Classification (CPC)"),

            num1 = gr.Slider(1, 5, step=1, value=2, label="Number of Key Words")
            num2 = gr.Slider(1, 5, step=1, value=2, label="Number of Classifications")
            #output = gr.Number(label="Sum")
            @gr.on(inputs=[num1, num2])
            def sum(a, b, c):
                return a + b + c

            gr.CheckboxGroup(["Google Patents", "Espacenet", "European Patent Office (EPO)", "DEPATISnet"], label="Databases", info="Which databases should be searched?", value="Google Patents"),
            
            files= gr.File(file_types=['.pdf'], label="Upload your pdf here.")

            button = gr.Button("Submit")
            

        with gr.Column() as sidebar_right:
            gr.Markdown("<p><h1>Output</h1></p>")
            result = gr.Textbox(label="Input")      

            keywords = gr.Textbox(label="Key Words", value="None") #New Value "<List of Key Words>"
            classes = gr.Textbox(label="Classifications", value="None") #New Value "<List of Classifications>"

            result.change(output_keywords, result, keywords) 
            result.change(output_classes, result, classes) 

            endresult = gr.Textbox(label="End Result", value="None")
            
            classes.change(patent_analysis_rest, [result, keywords, classes], endresult) #Does not matter if classes or results           

            with gr.Accordion(label= "Detailed Steps", open=False):   

                gr.Textbox(label="API OpenAI", value="Disconnected") #New Value "Connected"
                gr.Textbox(label="API Patent Database #1", value="Disconnected") #New Value "Connected"
                gr.Textbox(label="API Call #1", value="Disconnected") #% Schritte in Anzahl PDFs; New Value "Added n PDFs to the list"
                gr.Textbox(label="API Call #n", value="Disconnected")
                gr.Textbox(label="PDF List", value="No PDFs added yet") #New Value "xx PDFs added to the list."
                gr.Textbox(label="API Connection Vector Database", value="Disconnected") #New Value "Connected"
                gr.Textbox(label="Collection Vector Database", value="No PDFs added yet") #New Value "xx PDFs added to the collection of vector database."
                gr.Textbox(label="Compare Input with PDFs in Collection", value="...") #New Value "Top 5 PDFs ...."
            visibility=False
            if create_pdf:
                visibility = True
            pdf_file = gr.Button("Create PDF")    
            with gr.Row(visible=visibility):
                    outputs = "file"
                    gr.Button.click(create_pdf, inputs=[result], outputs=[outputs])
            
            button.click(patent_analysis, inputs=[files], outputs=[result])

                   
demo.launch()

# <a href="https://de.freepik.com/fotos-kostenlos/geschaeftsmann-haelt-gelbe-gluehbirne-mit-kopierraum-fuer-geschaeftsloesung-und-kreatives-denken-ideenkonzept-durch-3d-darstellung_26791662.htm#query=patente&position=5&from_view=search&track=sph&uuid=7931811d-4408-4f21-a68a-3450f7e46c8d">Bild von DilokaStudio</a> auf Freepik
