import threading
import time
import numpy as np
from datetime import timedelta

from src.config import settings
from src.services.model import SRTModel
from src.services.audio_processor import audio_queue
from src.services.subtitle_manager import add_subtitle
from src.services import logger

model = SRTModel(
    model=settings.MODEL,
    device=settings.DEVICE,
    compute_type=settings.COMPUTE_TYPE,
    download_root=settings.DOWNLOAD_ROOT
)


def process_audio():
    buffer = []
    segment_start_time = time.time()

    while True:
        audio_data = audio_queue.get()
        audio_chunk = np.frombuffer(audio_data, dtype=np.int16)
        buffer.append(audio_chunk)

        if len(buffer) >= 10:  # Каждые 10 пакетов
            audio_data = np.concatenate(buffer, axis=0)
            buffer = []
            end_time = time.time()

            text = model.transcribe(audio_data)

            if text.strip():
                add_subtitle(text, timedelta(seconds=segment_start_time), timedelta(seconds=end_time))

            segment_start_time = time.time()


process_thread = threading.Thread(target=process_audio)
process_thread.start()
