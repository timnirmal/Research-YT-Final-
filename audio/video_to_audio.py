import subprocess
import os
from pydub import AudioSegment


def convert_video_to_audio_ffmpeg(video_file, output_ext="wav"):
    """Converts video to audio directly using `ffmpeg` command
    with the help of subprocess module"""
    print("Converting video to audio")
    filename, ext = os.path.splitext(video_file)
    subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{filename}.{output_ext}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)

    print(f"Converted {video_file} to {filename}.{output_ext}")

    return f"{filename}.{output_ext}"


def wav_to_mono_flac(audio_file):
    """Converts audio to mono channel and flac format"""
    print("Converting to mono channel and flac format")

    song = AudioSegment.from_wav(audio_file)
    # save as audio_file excluded the extension
    filename, ext = os.path.splitext(audio_file)
    song = song.set_channels(1)
    song.export(f"{filename}.flac", format="flac")

    print(f"Converted {audio_file} to {filename}.flac")

    return f"{filename}.flac"
