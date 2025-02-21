import threading
import srt
import time

from src.config import settings

subtitles = []
subtitle_index = 1


def add_subtitle(text, start_time, end_time):
    global subtitle_index
    if text:
        subtitles.append(srt.Subtitle(index=subtitle_index, start=start_time, end=end_time, content=text))
        subtitle_index += 1


def save_subtitles():
    while True:
        with open(settings.SRT_OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(srt.compose(subtitles))
        time.sleep(1)


saving_thread = threading.Thread(target=save_subtitles, daemon=True)
saving_thread.start()
