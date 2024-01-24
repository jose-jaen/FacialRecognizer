import os

from dotenv import load_dotenv

# Get URL address
load_dotenv()
website = os.getenv('URL')
