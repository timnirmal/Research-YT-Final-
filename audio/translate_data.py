import os
import speech_recognition as sr
from pydub.silence import split_on_silence
from pydub import AudioSegment


# create a speech recognition object
r = sr.Recognizer()

def get_large_audio_transcription(path):
    # open the audio file using pydub
    sound = AudioSegment.from_wav(path)
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
                              # experiment with this value for your target audio file
                              min_silence_len=500,
                              # adjust this per requirement
                              silence_thresh=sound.dBFS - 14,
                              # keep the silence for 1 second, adjustable as well
                              keep_silence=500,
                              )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""

    # process each chunk
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")

        # print("Processing chunk " + str(i) + "/" + str(len(chunks)) + "...")
        # print("Chunk " + str(i) + " saved as " + chunk_filename + ".")
        # print(audio_chunk)

        # recognize the chunk

        with sr.WavFile(chunk_filename) as source:
            audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened, language="si-LK")
            except sr.UnknownValueError as e:
                # print("Error:", str(e))
                pass
            else:
                text = f"{text.capitalize()}. "
                # print(chunk_filename, ":", text)
                whole_text += text


    # return the text for all chunks detected
    return whole_text
