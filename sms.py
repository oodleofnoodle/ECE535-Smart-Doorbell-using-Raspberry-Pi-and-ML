import requests
from requests.auth import HTTPBasicAuth

USERNAME = "YOUR_CLICKSEND_USERNAME"
API_KEY = "YOUR_CLICKSEND_API_KEY"

def send_sms(to_number, message):
    url = "https://rest.clicksend.com/v3/sms/send"
    data = {
        "messages": [
            {
                "source": "python",
                "body": message,
                "to": to_number
            }
        ]
    }

    response = requests.post(
        url,
        json=data,
        auth=HTTPBasicAuth(USERNAME, API_KEY)
    )

    print("Status:", response.status_code)
    print("Response:", response.json())


# Example test message:
send_sms("+15555555555", "Test message from Python via ClickSend!")
