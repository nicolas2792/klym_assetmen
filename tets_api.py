import requests


url = "http://127.0.0.1:8000/predict/"
data = {
    "type": "sedan",
    "year": 2018,
    'fuel':'gas',
    "model": "camry",
    'transmission':'manual',
    "manufacturer": "toyota",
    "drive": "AWD",
    "odometer": 65800,
    'title_status':'clean',
    'cylinders':'8 cylinders',
    'poverty':12.4,
    'crashes':2345
}

response = requests.post(url, json=data)
print(response.json())