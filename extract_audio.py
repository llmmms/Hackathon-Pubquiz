import openai
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


azure_api_key_whisper = os.getenv('AZURE_OPENAI_API_KEY_WHISPER')
azure_endpoint_whisper = os.getenv('AZURE_OPENAI_ENDPOINT_WHISPER')

client = openai.AzureOpenAI(
    api_key=azure_api_key_whisper,
    azure_endpoint=azure_endpoint_whisper,
    azure_deployment="whisper",
    api_version="2023-09-01-preview",
)

def get_transcript(audio_file):
    if not os.path.exists(audio_file):
        audio_file = "./data/" + audio_file
    client.audio.with_raw_response
    return client.audio.transcriptions.create(
        file=open(audio_file, "rb"),
        model="whisper",
        language="de",
    ).text

audio_dir = Path("./PubAudio/")

transcripts = []

for audio_file in audio_dir.glob("*.mp3"):
    transcripts.append(get_transcript(str(audio_file)))
    with open("./PubTexts/" + audio_file.stem + ".txt", "w", encoding="utf-8") as f:
        f.write(get_transcript(str(audio_file)))
