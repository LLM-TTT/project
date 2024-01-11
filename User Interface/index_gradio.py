import gradio as gr
import PyPDF2

def extract_text_from_pdf(pdf_file):
    # Open the PDF file
    with open(pdf_file.name, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfFileReader(file)
        
        # Extract text from all pages
        text = ''
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            text += page.extractText()
    
    return text

def upload_file(files):
    file_paths = [file.name for file in files]
    
    # Extract text from PDF files
    extracted_texts = []
    for file in files:
        if file.type == "application/pdf":
            text = extract_text_from_pdf(file)
            extracted_texts.append(text)
        else:
            extracted_texts.append("Not a PDF file, cannot extract text.")
    
    return extracted_texts

with gr.Blocks() as demo:
    file_output = gr.File()
    upload_button = gr.UploadButton("Click to Upload a File", file_types=["application/pdf"], file_count="multiple")
    upload_button.upload(upload_file, upload_button, file_output)

demo.launch()

