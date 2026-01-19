import requests
import base64

# Read + encode image
with open('data/sample_xray.png', 'rb') as f:
    img_data = base64.b64encode(f.read()).decode()

# Test API
response = requests.post('http://localhost:9696/predict', json={'image': img_data})
print(response.json())
