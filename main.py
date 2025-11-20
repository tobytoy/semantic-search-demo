import os
import gdown
import subprocess
from dotenv import load_dotenv
load_dotenv()

url = os.getenv('DRIVE_URL')
output = "archive.zip"
gdown.download(url, output, quiet=False)
subprocess.run(['unzip', 'archive.zip'], check=True)
subprocess.run(["streamlit", "run", "app.py"])
