import subprocess
import threading
from src.config import settings
from src.services import logger


def start_new_stream():
    command = [
        'ffmpeg',
        '-loglevel', 'info',  # включаем подробное логирование для отладки
        '-i', settings.RTMP_URL,
        '-vf', f"subtitles={settings.OUTPUT_SRT_PATH}:reload=1",  # ключ reload=1 позволяет перезагружать файл субтитров
        '-c:v', 'libx264',  # Кодек для видео
        '-preset', 'veryfast',  # Параметры производительности
        '-maxrate', '3000k',
        '-bufsize', '6000k',
        '-pix_fmt', 'yuv420p',
        '-g', '50',
        '-c:a', 'copy',  # Копируем аудио без перекодирования
        '-f', 'mpegts',  # Используем MPEG-TS для UDP
        settings.OUTPUT_UDP_URL
    ]

    process = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    logger.info("A stream with subtitles has been started on %s", settings.OUTPUT_UDP_URL)

    threading.Thread(target=log_ffmpeg_output, args=(process,)).start()


def log_ffmpeg_output(process):
    while True:
        output = process.stderr.readline()
        if output == b"" and process.poll() is not None:
            break
        if output:
            print(output.decode().strip())


stream_thread = threading.Thread(target=start_new_stream)
stream_thread.start()
