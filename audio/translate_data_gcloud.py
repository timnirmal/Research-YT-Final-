import pandas as pd
from google.cloud import storage
from google.cloud import speech
from pydub import AudioSegment


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

    blob.chunk_size = 5 * 1024 * 1024  # Set 5 MB blob size
    blob.upload_from_filename(file_name)

    print('File upload complete')
    return


def google_transcribe_single(audio_file, bucket):
    # convert audio to text
    gcs_uri = 'gs://' + bucket + '/' + audio_file
    transcript = ''

    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)
    frame_rate = 44100

    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=5,
        max_speaker_count=20,
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=frame_rate,
        language_code='si-LK',
        # model='video',  # optional: specify audio source. This increased transcription accuracy when turned on
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        diarization_config=diarization_config,  # optional: Enable automatic punctuation
    )

    # Detects speech in the audio file
    operation = client.long_running_recognize(config=config, audio=audio)  # asynchronous
    response = operation.result(timeout=10000)

    for result in response.results:
        alternative = result.alternatives[0]
        print("-" * 20)
        print(alternative)
        print("Transcript: {}".format(alternative.transcript))
        print("Confidence: {}".format(alternative.confidence))

        for word_info in alternative.words:
            word = word_info.word
            start_time = word_info.start_time
            end_time = word_info.end_time

            print(f"Word: {word}, start_time: {start_time.total_seconds()}, end_time: {end_time.total_seconds()}")

        transcript += result.alternatives[0].transcript

    print(transcript)
    return transcript


def google_transcribe_single_df(audio_file, bucket, df):
    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=5,
        max_speaker_count=20,
    )


    # convert audio to text
    gcs_uri = 'gs://' + bucket + '/' + audio_file
    transcript = ''

    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)
    frame_rate = 44100

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=frame_rate,
        language_code='si-LK',
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        diarization_config=diarization_config,  # optional: Enable automatic punctuation
    )

    # Detects speech in the audio file
    operation = client.long_running_recognize(config=config, audio=audio)  # asynchronous
    response = operation.result(timeout=10000)

    for result in response.results:
        alternative = result.alternatives[0]
        print("-" * 20)
        print(alternative)
        print("Transcript: {}".format(alternative.transcript))
        print("Confidence: {}".format(alternative.confidence))

        for word_info in alternative.words:
            word = word_info.word
            start_time = word_info.start_time
            end_time = word_info.end_time

            # print(f"Word: {word}, start_time: {start_time.total_seconds()}, end_time: {end_time.total_seconds()}")
            # df with concat
            df = pd.concat([df, pd.DataFrame({'word': word, 'text': alternative.transcript, 'start_time': start_time.total_seconds(), 'end_time': end_time.total_seconds(), 'speaker': word_info.speaker_tag, 'confidence': alternative.confidence}, index=[0])], ignore_index=True)

        transcript += result.alternatives[0].transcript

    print(transcript)
    return transcript, df


def write_transcripts(transcript_file, transcript):
    f = open(transcript_file, "w", encoding="utf-8")
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

def get_large_audio_transcription(audio_file):
    bucket = "audio-store-audio-to-speach-reseach-1"
    #
    # # do only if file is .wav
    # prep_audio_file(audio_file)

    # upload audio file to storage bucket
    upload_blob(bucket, "", audio_file, audio_file)

    # create dataframe word, text, start_time, end_time, speaker
    df = pd.DataFrame(columns=['word', 'start_time', 'end_time', 'speaker', 'text', 'confidence'])

    # create transcript
    transcript, df = google_transcribe_single_df(audio_file, bucket, df)
    transcript_file = audio_file.split('.')[0] + '.txt'

    write_transcripts(transcript_file, transcript)
    print(f'Transcript {transcript_file} created')

    # remove confidence 0.0
    df = df[df['confidence'] != 0.0]
    # save df to csv
    df.to_csv(audio_file.split('.')[0] + '.csv', index=False)

    # remove audio file from bucket
    delete_blob(bucket, audio_file)

    return transcript, df
