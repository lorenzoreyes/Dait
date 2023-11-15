'''
Read the pdfs files of tickets
and convert them into txt files 
'''
import pytesseract, glob, re 
from pdf2image import convert_from_path

pdfs = glob.glob(r"*.pdf")

# convert pdfs images into txt files
for pdf_path in pdfs:
    pages = convert_from_path(pdf_path, 800)

    for pageNum,imgBlob in enumerate(pages):
        text = pytesseract.image_to_string(imgBlob,lang='spa')

        with open(f'{pdf_path[:-4]}_page{pageNum}.txt', 'w') as the_file:
            the_file.write(text)
