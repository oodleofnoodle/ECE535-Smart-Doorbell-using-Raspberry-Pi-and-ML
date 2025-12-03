import requests
from requests.auth import HTTPBasicAuth

USERNAME = "jessiewang@umass.edu"
API_KEY = "2ED477EE-40D6-8A8A-80C0-FD0DCA1C6270"

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


if __name__ == "__main__":
    # Example usage for manual testing. Update USERNAME, API_KEY, and phone number before running.
    send_sms("+19789239228", "ALERT: SUSPICIOUS PERSON AT DOOR")
