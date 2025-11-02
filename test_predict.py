import requests

url = "http://127.0.0.1:5000/predict"

data = {
      'gender': 'male',
      'major': 'IT',
      'admission_type': 'A',
      'financial_state': 'Average' 
}

response = requests.post(url, json=data)

print("Kết quả trả về từ server")
print(response.json())
