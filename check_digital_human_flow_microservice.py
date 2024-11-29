import base64
import json
import os
import urllib.request
import uuid

import requests

# https://gist.github.com/novwhisky/8a1a0168b94f3b6abfaa
# test_audio_base64_str = "UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA"

uid = str(uuid.uuid4())
file_name = uid + ".wav"

urllib.request.urlretrieve(
    "https://github.com/intel/intel-extension-for-transformers/raw/main/intel_extension_for_transformers/neural_chat/assets/audio/sample.wav",
    file_name,
)

with open(file_name, "rb") as f:
    test_audio_base64_str = base64.b64encode(f.read()).decode("utf-8")
os.remove(file_name)

endpoint = "http://localhost:8765/v1/digital_human_flow"
inputs = {"audio": test_audio_base64_str}
response = requests.post(url=endpoint, data=json.dumps(inputs), proxies={"http": None})
# print(response.json())
print(response)


with open("out.mp4", 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)