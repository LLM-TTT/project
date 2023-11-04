from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
import datetime
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  
openai.api_key = os.environ['OPENAI_API_KEY']


llm_model = "gpt-3.5-turbo"

chat = ChatOpenAI(temperature=0.0, model=llm_model)