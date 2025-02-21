import os

from pydantic_settings import BaseSettings


def get_exclude_phrases():
    path = "exclude_phrases.txt"

    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f]


class Settings(BaseSettings):
    RTMP_URL: str
    OUTPUT_UDP_URL: str
    MODEL: str
    DEVICE: str
    COMPUTE_TYPE: str
    DOWNLOAD_ROOT: str
    SAMPLE_RATE: int = 16000
    SRT_OUTPUT_PATH: str = "subtitles.srt"

    class Config:
        env_file = ".env"


settings = Settings()
exclude_phrases = get_exclude_phrases()
