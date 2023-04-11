from PyPDF2 import PdfReader

def pdftotext():
    reader = PdfReader("docs/pdf1.pdf")
    number_of_pages = len(reader.pages)
    context = " "
    for i in range(number_of_pages):
        if i==0:
            page = reader.pages[i]
            text = page.extract_text()  
            context += text

    return context 


#TODO
'''
create a func to process 3 pdfs, i/o in terminal, different models, training process?
'''