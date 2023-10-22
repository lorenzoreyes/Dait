import pytesseract
from pdf2image import convert_from_path
import glob
import pandas as pd 

pdfs = glob.glob(r"tres.pdf")

for pdf_path in pdfs:
    pages = convert_from_path(pdf_path, 800)

    for pageNum,imgBlob in enumerate(pages):
        text = pytesseract.image_to_string(imgBlob,lang='eng')

        with open(f'{pdf_path[:-4]}_page{pageNum}.txt', 'w') as the_file:
            the_file.write(text)


response = text.split('\n')

fecha = [i for i in response if 'Fecha' in i]
productos = [i for i in response if '-(' in i]
qxp = [i for i in response if 'u x' in i]
quantity = [float(i.split(' u')[0].replace(',','.').replace(' ','')) for i in qxp]
price = [float(i.split('x ')[-1].replace(',','.').replace(' ','')) for i in qxp]
paid = [quantity[i] * price[i] for i in range(len(quantity))]

data = pd.DataFrame(productos,columns=['Productos'])
data['Cantidad'] = quantity
data['Precio'] = price
data['Pagado'] = data['Cantidad'] * data['Precio']
data['Total'] = data['Pagado'].sum()
data['Fecha'] = fecha[0]

print(data)


