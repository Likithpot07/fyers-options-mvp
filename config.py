import os
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("FYERS_CLIENT_ID")
ACCESS_TOKEN = os.getenv("FYERS_ACCESS_TOKEN")
SECRET_KEY = os.getenv("FYERS_SECRET_KEY")
REFRESH_TOKEN = os.getenv("FYERS_REFRESH_TOKEN")