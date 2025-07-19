import os

from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()
account_sid = os.getenv("ACCOUNT_SID")
auth_token = os.getenv("AUTH_TOKEN")
client = Client(account_sid, auth_token)

message = client.messages.create(
    from_='whatsapp:+14155238886',
    body="Hello, this is a test message from Python. Thanks for using Twilio!",
    to='whatsapp:+14698415757'
)

print(message.sid)
