#!C:\Users\ruhmt\AppData\Local\Programs\Python\Python311\python.exe
import streamlit as st
import PyPDF2

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