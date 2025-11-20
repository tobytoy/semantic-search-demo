import os
import gdown
import subprocess
from dotenv import load_dotenv
load_dotenv()
output = "archive.zip"
if not os.path.exists(output):
    url = os.getenv('DRIVE_URL')
    gdown.download(url, output, quiet=False)
    subprocess.run(['unzip', 'archive.zip'], check=True)
subprocess.run(["streamlit", "run", "app.py"])
