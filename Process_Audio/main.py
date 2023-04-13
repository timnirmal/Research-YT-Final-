import subprocess
import os
import sys
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from lib.video_to_audio import convert_video_to_audio_ffmpeg
from lib.Translate import get_large_audio_transcription
from lib.process_data import process_data

file_name = "Ex05.mkv"

convert_video_to_audio_ffmpeg(file_name)

transcripted_text = get_large_audio_transcription(file_name[:-4] + ".wav")
print(transcripted_text)

# save text to file
with open("recognized.txt", "w", encoding="utf-8") as f:
    f.write(transcripted_text)

processed_data = process_data("recognized.txt")
print(processed_data)






