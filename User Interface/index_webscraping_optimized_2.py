import gradio as gr
import PyPDF2
import time
import os
from bs4 import BeautifulSoup as bs
import openai
#import OpenAI
import requests
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# Closing all open ports
gr.close_all()

# Loading OpenAI API Key and Google Patent API Key
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']
patent_api_key = os.environ['GOOGLE_PATENT_API_KEY']

# Defining LLM Model
llm_model = "gpt-4"
client = openai
def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}] #role: define the role of the llm; conent: how the llm should act
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, #creativity range 0..1
    )
    return response.choices[0].message.content

# Loading PDF file from user input and extracting first page containing the abstract
def input_analysis(file, progress=gr.Progress()):
    if file is not None:
        #Reading PDF file
        progress(0.2, desc="Processing file")
        pdf_reader = PyPDF2.PdfReader(file)
        # Extracting content
        content = pdf_reader.pages[0].extract_text()
        if content == "":
            raise gr.Error("The file seems to be invalid. Please check your file or try another Pdf!")
        progress(0.3, desc="Analyzing file")
    return content

# LLM Prompt generating Key Words on base of the PDF Input from User
def output_keywords(content, n, progress=gr.Progress()):
    progress(0, desc="Generating Key Words...")
    keyword_prompt = f"""
    The following abstract descripes a concept for a novel invention:\
    ```{content}```\
    Name {n} key words based on this abstract, that I can use for the search in a patent database. \
    Optimize the key words to get back more results. Result as python string.
    """

    response_keywords = get_completion(keyword_prompt)

    return response_keywords

# LLM Prompt generating Classifications on base of the PDF Input from User
def output_classes(content, n, progress=gr.Progress()):
    progress(0, desc="Generating Classifications...")
    classes_prompt = f"""
        The following abstract descripes a concept for a novel invention:\
        ```{content}```\
        Name {n} CPC classifications based on this abstract, that I can use for the search in a patent database. \
        Please give me a python string for the codes of the {n} most relevant \
        CPC classifications to a possible patent. 
        """
    
    response_classes = get_completion(classes_prompt)
    
    return response_classes

# Connecting to database
def get_database_connection():
    uri = os.environ['DATABASE_URI']
    client = MongoClient(uri, server_api=ServerApi('1'))
    DB_NAME = "llm-ttt"
    COLLECTION_NAME = "pdfresults"
    MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]
    return MONGODB_COLLECTION

# Clearing vector database
def clear_db():
    get_database_connection()
    x = MONGODB_COLLECTION.delete_many({})
    delete = str(x.deleted_count) + " documents deleted."
    return delete

# Continue with API Calls, vectorizing and vector database handling
def patent_analysis(content, response_keywords, response_classes, progress=gr.Progress()):
    
    # transforming LLM response into keyword list
    keywords_list = [keyword.strip('"') for keyword in response_keywords.split(", ")]
    print(keywords_list)

    # cast the results (classifications) from string to list
    class_list = [classification.strip('"') for classification in response_classes.split(", ")]
    print(class_list)

    # initializing base vars for the following loop
    progress(0.5, desc="Researching patents")
    patent_api_key = os.environ['GOOGLE_PATENT_API_KEY']
    count = 0
    patent_base_url = "https://patentimages.storage.googleapis.com/" #just to complete the url
    patent_data = {}

    # Loop for multiple Google Patents API calls with Key Words
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

    # Parsing API answer .json and adding the patent ID to the patent_data dict, avoiding duplicates
    for cluster in data["results"]["cluster"]:
        for result in cluster["result"]:
            patent_id = result["patent"]["publication_number"]
            if patent_id not in patent_data.keys():
                patent_data[patent_id] = {
                    "pdf": result["patent"]["pdf"],
                }

    # Scraping complete patent data (title, abstract, description, claims)
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
        
        # Adding scraped content do patent_data dict
        patent_data[patent_id].update({
            "title": title,
            "abstract": abstract,
            "description": description,
            "claims": claims
        })

    # Converting patent data into a required format to perform a vector search
    patent_list = []
    for patent_id, data in patent_data.items():
        page_content = f"{data['title']} {data['abstract']} {data['description']} {data['claims']}"
        metadata = {"patent_id": patent_id}
        patent_list.append(Document(page_content=page_content, metadata=metadata))

    # Clearing db before adding new data, to avoid any distortion of results
    clear_db()

    # Login MongoDB with User and specific database
    uri = "mongodb+srv://timmey:faB8MFdyyb7zWvVr@llm-ttt.8kqrnka.mongodb.net/?retryWrites=true&w=majority"

    get_database_connection()
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"      

    # insert the documents in MongoDB Atlas with their embedding
    vector_search = MongoDBAtlasVectorSearch.from_documents(
        documents=patent_list,
        embedding=OpenAIEmbeddings(disallowed_special=()),
        collection=MONGODB_COLLECTION,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )

    # waiting for the db to save
    time.sleep(5)

    # Performing a similarity search with Score between the embedding of the query and the embeddings of the documents
    progress(0.9, desc="Compare the patents")
    query = str(content)

    results = vector_search.similarity_search_with_score(
        query=query,
        k=20, #Output for the top 20 results
    )

    # Formatting vector search result for further usage
    vector_scoring = {}
    for result in results:
        vector_scoring[result[0].metadata['patent_id']] = result[1]
    
    # Building LLM similarity scoring prompt
    comparison_prompt = f"""The following texts are abstracts from patent specifications. Your task is to compare the "Testing Abstract" to all the others. 
    It is important that you focus on comparing the concepts that the abstracts describe, not the way they are written. 
    Rank the remaining abstracts on how well they match with the Testing Abstract by giving them a rating from 0 to 10 points. 
    0 meaning they have absolutely nothing in common and 10 meaning they basically describe the exact same idea.
    Your output should be a python dictionary with the title "comparison", each element hast the Abstract number as key and the rating as value.
    I want to convert your output string to an actual dictionary, so make sure the formatting is right.

    Testing Abstract: "{content}"
    """

    # Adding patent abstracts to the prompt
    for patent_id in vector_scoring.keys():
        # Check if there is an abstract for the patent
        if patent_id in patent_data and patent_data[patent_id]["abstract"]:
            comparison_prompt += f'{patent_id}: "{patent_data[patent_id]["abstract"]}"\n'

    response = get_completion(comparison_prompt)

    # Formatting LLM output
    llm_scoring_raw = eval(response.replace("comparison = ",""))

    # Calculating final scoring results (combining vector scoring with llm scoring)
    def transform_ratings(ratings, new_min=0, new_max=10):
        # Determine the smallest and largest value in the original dictionary
        old_min, old_max = min(ratings.values()), max(ratings.values())
        transformed_ratings = {}
        for key, value in ratings.items():
            # Apply the transformation with dynamic old and new ranges
            transformed_value = ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
            transformed_ratings[key] = transformed_value
        return transformed_ratings

    def delete_entries_with_zero(ratings):
        # Create a new dictionary without the entries with the value 0
        cleaned_dict = {key: value for key, value in ratings.items() if value != 0}
        return cleaned_dict

    def calculate_average_and_unite(dict_a, dict_b):
        combined_dict = {}
        # Union of keys from both dictionaries
        all_keys = set(dict_a.keys()) | set(dict_b.keys())
        
        for key in all_keys:
            values = []
            if key in dict_a:
                values.append(dict_a[key])
            if key in dict_b:
                values.append(dict_b[key])
            # Calculate the average if the key is present in both dictionaries
            combined_dict[key] = sum(values) / len(values)
        
        # Sort the dictionary from high to low based on the values
        sorted_combined_dict = dict(sorted(combined_dict.items(), key=lambda item: item[1], reverse=True))
        
        return sorted_combined_dict

    def transform_and_calculate(dict_a, dict_b):
        a = transform_ratings(dict_a)
        b = transform_ratings(dict_b)
        return calculate_average_and_unite(a, b)

    llm_scoring = delete_entries_with_zero(llm_scoring_raw)
    final_scoring = transform_and_calculate(llm_scoring, vector_scoring)

    final_scoring_patent_ids = list(final_scoring.keys())

    # Formatting final scoring results for user output
    final_scoring_formatted = "Ergebnis:\n\n"
    counter=1
    for patent_id in final_scoring_patent_ids:
        final_scoring_formatted += "#" + str(counter) + ": " + patent_data[patent_id]["title"] + "\n" + "https://patentimages.storage.googleapis.com/" + patent_data[patent_id]["pdf"] + "\n\n"
        counter+=1

    return final_scoring_formatted

with gr.Blocks(theme=gr.themes.Glass(primary_hue=gr.themes.colors.zinc, secondary_hue=gr.themes.colors.gray, neutral_hue=gr.themes.colors.gray)) as demo:
    gr.Markdown("# Patent Pete")
    gr.Markdown("<p style='font-size:16px;'>Hi, I'm Pete your assistance for patent researchs. I help you with finding out, if your patent already exists. Let's start with uploading your idea!</p>")
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
            
            classes.change(patent_analysis, [result_output, keywords, classes], endresult) #It does not matter if you choose classes or keywords from above

            clear_button = gr.Button("New Research")
            clear_button.click(clear_db,outputs=[endresult])
            
            button.click(input_analysis, inputs=[files], outputs=[result_output])
                   
demo.launch(enable_queue=True)