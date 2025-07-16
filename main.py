import os

from dotenv import load_dotenv
from flask import Flask
from twilio.rest import Client

# chatbot logic
def bot():
    load_dotenv()
    auth_token = os.getenv("AUTH_TOKEN")
    account_sid = os.getenv("ACCOUNT_SID")

    client = Client(account_sid, auth_token)

    message = client.messages.create(
        from_="whatsapp:+14155238886",
        body="Hello, there!",
        to="whatsapp:+15005550006",
    )

    return print(message.body)

bot()