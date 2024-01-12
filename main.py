import pafy
import os
from transformers import pipeline

audio_file_path = os.getcwd() + r'/save/audio.mp3';

video = pafy.new("https://youtu.be/1j0X9QMF--M?si=181hP0WKYM5MPEak") #Replace URL here
audio = video.getbestaudio()

audio.download(filepath = audio_file_path)

model = "facebook/wav2vec2-large-960h-lv60-self"  # speech to text

# speech to text
pipe = pipeline(model=model)
text = pipe(audio_file_path, chunk_length_s=10)

# save text
text_file = open("original_text.txt", "w")
n = text_file.write(text["text"])
text_file.close()

# read article
text_article = open("original_text.txt", "r").read()

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print(summarizer(text_article, max_length=100, min_length=50, do_sample=False))
