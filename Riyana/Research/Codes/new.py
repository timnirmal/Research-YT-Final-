import os
import subprocess
from pytube import YouTube
from google.cloud import storage
from google.cloud import speech
# from google.cloud.speech import enums
# from google.cloud.speech import types
from textblob import TextBlob

a_file = "Ex05.wav"

from pydub import AudioSegment

song = AudioSegment.from_wav(a_file)
song.export("testme.flac", format="flac")

audio_file = "testme.flac"


def prep_audio_file(audio_file):
    """
    This function makes sure audio file meets requirements for transcription:
    - Must be mono
    """
    # modify audio file
    sound = AudioSegment.from_wav(audio_file)
    sound = sound.set_channels(1)

    # can be useful to resample rate to 16000. google recommends to not do this but can be used to tune
    # sound = sound.set_frame_rate(16000)
    sound.export(audio_file, format="wav")
    return


def upload_blob(bucket_name, audio_path, audio_file, destination_blob_name):
    """Uploads a file to the bucket.
    Inputs:
        # bucket_name = "your bucket name"
        # audio_path = "path to file"
        # audio_file = "file name"
        # destination_blob_name = "storage object name"
    """
    file_name = audio_path + audio_file

    # upload audio file to storage bucket
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.chunk_size = 5 * 1024 * 1024 # Set 5 MB blob size
    blob.upload_from_filename(file_name)

    print('File upload complete')
    return


def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket.
    Inputs:
        # bucket_name = "your bucket name"
        # blob_name = "storage object name"
    """
    storage_client = storage.Client()

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()

    print(f'Blob {blob_name} deleted')
    return

def speech_to_text(
        config: speech.RecognitionConfig,
        audio: speech.RecognitionAudio,
) -> speech.RecognizeResponse:
    client = speech.SpeechClient()

    # Synchronous speech recognition request
    response = client.recognize(config=config, audio=audio)

    return response


def print_response(response: speech.RecognizeResponse):
    for result in response.results:
        print_result(result)


# def print_result(result: speech.SpeechRecognitionResult):
#     best_alternative = result.alternatives[0]
#     print("-" * 80)
#     print(f"language_code: {result.language_code}")
#     print(f"transcript:    {best_alternative.transcript}")
#     print(f"confidence:    {best_alternative.confidence:.0%}")

def print_result(result: speech.SpeechRecognitionResult):
    best_alternative = result.alternatives[0]
    print("-" * 80)
    print(f"language_code: {result.language_code}")
    print(f"transcript:    {best_alternative.transcript}")
    print(f"confidence:    {best_alternative.confidence:.0%}")
    print("-" * 80)
    for word in best_alternative.words:
        start_s = word.start_time.total_seconds()
        end_s = word.end_time.total_seconds()
        print(f"{start_s:>7.3f} | {end_s:>7.3f} | {word.word}")


def simple_speech_to_text(audio_file):
    # config = speech.RecognitionConfig(language_code="en")

    config = speech.RecognitionConfig(
        language_code="en",
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        audio_channel_count=2,

    )
    # audio = speech.RecognitionAudio(
    #     uri="gs://cloud-samples-data/speech/brooklyn_bridge.flac",
    # )

    # audio_file to speech RecognitionAudio
    with open(audio_file, "rb") as audio_file:
        content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)

    response = speech_to_text(config, audio)
    print_response(response)

    # 4/0AVHEtk5Y2fRj9VtkzUhy7MW1mrAKLdr2lbszi_hqp8yWpy1vZFHpS3SmbSdjQdkhNRy9dg


def transcribe_gcs(gcs_uri):
    """Asynchronously transcribes the audio file specified by the gcs_uri."""
    from google.cloud import speech

    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        # sample_rate_hertz=16000,
        language_code="en-US",
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation to complete...")
    response = operation.result(timeout=90)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        print("Transcript: {}".format(result.alternatives[0].transcript))
        print("Confidence: {}".format(result.alternatives[0].confidence))


transcribe_gcs("gs://cloud-samples-data/speech/brooklyn_bridge.flac")


def transcribe_audio_file(audio_file):
    from google.cloud import speech

    client = speech.SpeechClient()

    with open(audio_file, "rb") as audio_file:
        content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        # sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        print("Transcript: {}".format(result.alternatives[0].transcript))
        print("Confidence: {}".format(result.alternatives[0].confidence))

    print(response)

    # print(response.results[0].alternatives[0].transcript)
    # print(response.results[0].alternatives[0].confidence)
    # print(response.results[0].alternatives[0].words[0].start_time)
    # print(response.results[0].alternatives[0].words[0].end_time)
    # print(response.results[0].alternatives[0].words[0].word)
    # print(response.results[0].alternatives[0].words[0].confidence)


def google_transcribe_single(audio_file, bucket):
    # convert audio to text
    gcs_uri = 'gs://' + bucket + '/' + audio_file
    transcript = ''

    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)
    frame_rate = 44100

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=frame_rate,
        language_code='si-LK',
        # model='video',  # optional: specify audio source. This increased transcription accuracy when turned on
        enable_automatic_punctuation=True)  # optional: Enable automatic punctuation

    # Detects speech in the audio file
    operation = client.long_running_recognize(config=config, audio=audio)  # asynchronous
    response = operation.result(timeout=10000)

    for result in response.results:
        transcript += result.alternatives[0].transcript

    print(transcript)
    return transcript


def write_transcripts(transcript_file, transcript):
    f = open(transcript_file,"w", encoding="utf-8")
    f.write(transcript)
    f.close()
    return

def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket.
    Inputs:
        # bucket_name = "your bucket name"
        # blob_name = "storage object name"
    """
    storage_client = storage.Client()

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()

    print(f'Blob {blob_name} deleted')
    return

# transcribe_audio_file("testme.flac")

bucket = "audio-store-audio-to-speach-reseach-1"
audio_file = "Ex05.wav"

# do only if file is .wav
prep_audio_file(audio_file)

# # upload audio file to storage bucket
upload_blob(bucket, "", audio_file, audio_file)

# create transcript
transcript = google_transcribe_single(audio_file, bucket)
transcript_file = audio_file.split('.')[0] + '.txt'

write_transcripts(transcript_file, transcript)
print(f'Transcript {transcript_file} created')


# remove audio file from bucket
delete_blob(bucket, audio_file)