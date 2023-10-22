import pytesseract, glob, re 
from pdf2image import convert_from_path
import pandas as pd
import datetime as dt

pdfs = glob.glob(r"*.pdf")
txt = pdfs.copy()

# convert pdfs images into txt files
for pdf_path in pdfs:
    pages = convert_from_path(pdf_path, 800)

    for pageNum,imgBlob in enumerate(pages):
        text = pytesseract.image_to_string(imgBlob,lang='spa')

        with open(f'{pdf_path[:-4]}_page{pageNum}.txt', 'w') as the_file:
            the_file.write(text)
            
            
txt = glob.glob(r"*.txt") 

# read txt & convert it into excel format           
def txt_to_data(txt):
    data = pd.read_fwf(txt,delim_whitspaace=True)
    lista = data[data.columns[0]].to_list()
    fecha = [i for i in lista if 'Fecha' in i]  
    productos = [i.split('(')[0] for i in lista if re.search(r'~\(', i) or re.search(r'-\(', i)]
    qxp = [i for i in lista if 'u x' in i]
    quantity = [float(i.split(' u')[0].replace('—','').replace(',','.').replace(' ','').replace('-','').replace('~','').replace('|','').replace("'","")) for i in qxp]
    price = [float(i.split('x ')[-1].replace(',','.').replace(' ','')) for i in qxp]
    paid = [quantity[i] * price[i] for i in range(len(quantity))]
    data = pd.DataFrame(productos,columns=['Productos'])
    data['Cantidad'] = quantity
    data['Precio'] = price
    data['Pagado'] = data['Cantidad'] * data['Precio']
    data['Total'] = data['Pagado'].sum()
    data['Fecha'] = fecha[0].replace('Fecha ','').replace(' Hora','').replace('!','1').replace('99','09')
    data['Fecha'] = [dt.datetime.strptime(i, '%d/%m/%Y %H:%M:%S') for i in data['Fecha']] 
    
    return data[['Fecha','Productos', 'Cantidad', 'Precio', 'Pagado', 'Total']]

data = txt_to_data(txt[0])

for i in range(1,len(txt)):
    data = pd.concat([data,txt_to_data(txt[i])]).sort_index()

# limpieza
data['Productos'] = [i.split('(')[0][:-3] for i in data['Productos'].to_list()]
data = data.sort_values(by='Fecha',ascending=False)
data.to_csv('data.csv')

print(data)
