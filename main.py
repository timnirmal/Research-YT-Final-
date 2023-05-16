import re

import pandas as pd

from audio.video_to_audio import convert_video_to_audio_ffmpeg, wav_to_mono_flac
from audio.translate_data_gcloud import get_large_audio_transcription
from audio.process_data import process_data, process_data_df
from audio.process_with_timestamp import process_with_timestamp

from video.model import predict

from lib.sync_av import sync_audio_and_video

from audio.hate_5 import predict_from_best
# from audio.hate_binary import predict_from_best
from audio.hate_polarity import analyze_sentiment

from multiprocessing import Process

from sentiment.sentiment_5 import predict_sentiment_sentece_from_best_df

from sentiment.sentiment_word import calculate_sentiment_df

from lib.classification import classify, cluster_model

# import ray
#
# ray.init()




#
# ############################################## Audio ##############################################################

# print("Converting video to audio")
#
# # convert video to audio
# convert_video_to_audio_ffmpeg(video_file)
#
# wav_to_mono_flac(video_file[:-4] + ".wav")
# @ray.remote
def processing_audio():
    print('Processing Audio Started...')

    print("Transcribing audio")

    transcripted_text, df = get_large_audio_transcription(video_file[:-4] + ".flac")

    print(transcripted_text)
    print(df)

    return transcripted_text, df

    # save csv
    # df.to_csv("recognized.csv", index=False)

    # save transcripted text to file
    with open("recognized.txt", "w", encoding="utf-8") as f:
        f.write(transcripted_text)

    # processed_data = process_data("recognized.txt")
    # print(processed_data)

    new_df = process_with_timestamp(df)
    print(new_df)

    new_df.to_csv("recognized_processed.csv", index=False)

    print('Processing Audio Ended.')


def processing_audio_hate(new_df):
    # predict hate for each unique text and add that to all rows with same text
    def predict_hate(df):
        # get unique text
        unique_text = df["text"].unique()

        # predict hate for each unique text
        for text in unique_text:
            # get all rows with same text
            rows = df[df["text"] == text]

            # get first row
            row = rows.iloc[0]

            # get hate
            hate = predict_from_best(row["text"])

            # add hate to all rows with same text
            df.loc[df["text"] == text, "hate"] = hate

        return df

    # predict hate
    new_df = predict_hate(new_df)
    # save csv
    new_df.to_csv("recognized_processed_hate.csv", index=False)


############################################## Video ##############################################################
# @ray.remote
def processing_video():
    print('Processing Video Started...')

    # predict
    predict(video_file) # filtered_frames.csv


def merge_audio_and_video():
    # load csv
    video_df = pd.read_csv("filtered_frames.csv")
    # audio_df = pd.read_csv("Ex05.csv")
    audio_df = pd.read_csv("recognized_processed_hate.csv")

    print(video_df.head(10))
    print(audio_df.head(10))

    merged_df = sync_audio_and_video(audio_df, video_df)

    print(merged_df.head(10)) # merged.csv

    merged_df.to_csv("merged_df.csv", index=False)

    print(merged_df.head(10))

    print('Processing Video Ended.')

    return merged_df


############################################## Sentiment ##############################################################

def processing_sentiment():
    # sentiment = analyze_sentiment(
    #     " elina දැන් උත්තර දෙන මං කැමති නෑ එතකොට නෑ දැන් ප්‍රේක්ෂකයෝ අපේ රටේ විශාල ප්‍රේක්ෂක පිරිසක් අපේ ලංකා කණ්ඩායම ගැන ලොකු බලාපොරොත්තුවකින් හිටියා අපිත් එහෙම ඉතින් ඒ වගේ දේවල් ගිලිහී ගෙන යද්දී එතකොට මොකද්ද ඇත්තෙන්ම ප්‍රේක්ෂකයාගේ ක්‍රීඩකයන් ක්‍රීඩා ගාථා නිදන් ග්‍රහණය කරන්න ඉතින් ඒක නිසා ඒක ඒ විදිහට තමයි බලාපොරොත්තුවෙන් වෙන්නේ නිර්මාණය ක්‍රීඩකයෝ කලකිරීමකින් ද පීඩා කරන්න ඔබ ඔය හිතට එකඟව කතා කරන්න ඔබතුමා ඉදිරියේදී ගන්න ක්‍රියාමාර්ග ගමගේ භාජනය කතාබහ කළ ඒගොල්ලොන්ට අවශ්‍ය දේ යනු ඕනෑම තීරණය කරන්නේ ඔබ සතුටු වෙනවාද යන්න ක්‍රීඩකයෝ කෙල්ලන්ට පීඩා කරපු ආකාරය ගැන පරාජය වෙලා ඔයා ඔයාගේ ප්‍රශ්නයක් අහන්නේ සම්බන්ධ මට හරි කනගාටුයි තරගය පරාජය ළමා සතුනට ගහණයක මං දන්නවා බුද්ධිමත් මාධ්‍යවේදියෙක් විදිහට නොකළ යුතු දෙයක්")
    #
    # print(sentiment)
    #
    # # # load csv
    # # df = pd.read_csv("Process_Audio/New_Hate_Sentiment/Sinhala_Singlish_Hate_Speech.csv")
    # #
    # # # process text df
    # # df = process_data_df(df, "Phrase")
    # #
    # # print(df.head(10))
    #
    # # save df
    # # df.to_csv("Process_Audio/New_Hate_Sentiment/Sinhala_Singlish_Hate_Speech_Processed.csv", index=False)
    #
    # df = pd.read_csv("Process_Audio/New_Hate_Sentiment/Sinhala_Singlish_Hate_Speech_Processed.csv")
    #
    # # remove null rows
    # df = df.dropna()
    #
    # # remove duplicates
    # df = df.drop_duplicates()
    #
    # # if Text_cleaned column has whitespaces, remove them
    # df['Text_cleaned'] = df['Text_cleaned'].str.strip()
    #
    # # remove null rows
    # df = df.dropna()
    #
    # # remove <null> rows
    # df = df[df['Text_cleaned'] != '<null>']
    #
    # # save df
    # df.to_csv("Sinhala_Singlish_Hate_Speech_Processed.csv", index=False)
    pass








if __name__ == '__main__':
    video_file = "Ex05.mkv"

    p1 = Process(target=processing_audio(video_file))
    p2 = Process(target=processing_video(video_file))
    p1.start()
    p2.start()
    #
    # p1.join()
    # p2.join()
    #
    # print("Done!")
    #
    # processing_audio_and_video_and_hate_for_audio()


    # @ray.remote
    # def func1():
    #     # print 10 to 20
    #     for i in range(10, 20):
    #         print(i,"1")
    #
    # @ray.remote
    # def func2():
    #     # print 1 to 10
    #     for i in range(10):
    #         print(i,"2")

    # Execute func1 and func2 in parallel.
    # ray.get([processing_audio.remote(), processing_video.remote()])
    #
    # print("Done!")
    #
    # df = merge_audio_and_video()
    #
    # df = predict_sentiment_sentece_from_best_df(df)
    #
    # print(df.head(10))
    #
    # df = calculate_sentiment_df(df)
    # # TODO: process the lexicon
    # print(df.head(10))
    #
    # df = cluster_model(df)
    #
    # # save df
    # df.to_csv("recognized_processed_sentiment_word.csv", index=False)
    #
    # classify(df)

    processing_audio()

