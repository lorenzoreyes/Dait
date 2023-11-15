import requests, re, json
import pandas as pd
from bs4 import BeautifulSoup

headers = {'User-agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.61 Safari/537.36'}
html_tag_pattern = r'<[^>]+>'
keys = ['description', 'quantity', 'price', 'total']
url = 'https://mifactura.napse.global/mf/pq1rt7/Y2VuY29zdWRfNjY5XzNfNl8wNjY5MDAzMDE5NjIzMTAzMDE5MjQ='

def read_ticket(url):
    response = requests.get(url, headers)
    soup =  BeautifulSoup(response.content, 'html.parser')
    contents = soup.contents[5]
    detail = soup.find_all("tr", {"class": "font table-full-alt"})
    detail = re.sub(html_tag_pattern, '', str(detail))
    # description, quantity, price & total
    detail = detail[3:-2].split('\n\n')
    data = dict(zip(keys,detail))
    data = pd.DataFrame([data])
    data['quantity'] = [float(i.replace(',','.')) for i in data['quantity']]
    data['price'] = [float(i.replace(',','.').replace('$ ','')) for i in data['price']]
    data['total'] = [float(i.replace(',','.').replace('$ ','')) for i in data['total']]
    emisor = soup.find_all("div", {"class": "company-name"})
    emisor = ''.join(re.sub(html_tag_pattern, '', str(emisor)))
    datos = soup.find_all("td", {"class": "datos left"})
    date = re.sub(html_tag_pattern, '', str(datos)).split('Emisión: ')[1].split(',')[0]
    data['Fecha'] = date
    data['Origen'] = emisor[1:-1]
    data = data[['Fecha','description','quantity','price','total','Origen']]
    data.columns = ['Fecha','Producto','Cantidad','Precio','Total','Origen']
    return data 

data = read_ticket(url)
print(data)
data.to_csv('vea.csv')
