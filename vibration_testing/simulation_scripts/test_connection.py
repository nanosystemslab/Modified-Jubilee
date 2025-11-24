import requests

JUBILEE_IP = "192.168.1.8"  # Your IP here
url = f"http://{JUBILEE_IP}/rr_status"

try:
    response = requests.get(url, timeout=5)
    print(f"Connection successful! Status code: {response.status_code}")
    print(f"Response: {response.text[:200]}...")
except Exception as e:
    print(f"Connection failed: {e}")