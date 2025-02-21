import subprocess
import queue
import threading
from src.config import settings
from src.services import logger

audio_queue = queue.Queue()


def capture_audio():
    """
    Capturing audio from an RTMP stream
    """

    command = [
        'ffmpeg',
        '-loglevel', 'quiet',
        '-i', settings.RTMP_URL,
        '-vn',  # no video
        '-f', 's16le',  # raw PCM signed 16-bit little-endian samples
        '-ar', str(settings.SAMPLE_RATE),  # sampling rate
        '-ac', '1',  # mono audio
        'pipe:1'  # output to stdout
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1024)
    logger.info("Audio capture from the stream has started %s", settings.RTMP_URL)

    while True:
        audio_data = process.stdout.read(1024)
        if not audio_data:
            break
        audio_queue.put(audio_data)


capture_thread = threading.Thread(target=capture_audio)
capture_thread.start()
