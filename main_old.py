import pandas as pd

from audio.video_to_audio import convert_video_to_audio_ffmpeg
from audio.translate_data import get_large_audio_transcription
from audio.process_data import process_data

from video.model import predict

video_file = "Ex05.mkv"

# convert video to audio
convert_video_to_audio_ffmpeg(video_file)

transcripted_text = get_large_audio_transcription(video_file[:-4] + ".wav")
print(transcripted_text)

# save text to file
with open("recognized.txt", "w", encoding="utf-8") as f:
    f.write(transcripted_text)

processed_data = process_data("recognized.txt")
print(processed_data)

# list to dataframe
df = pd.DataFrame(processed_data, columns=['text', 'start', 'end'])




# predict
predict(video_file)