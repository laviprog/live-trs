import os

from pydantic_settings import BaseSettings


def get_exclude_phrases():
    path = "exclude_phrases.txt"

    if not os.path.exists(path):
        return []

    with open(path, 'r') as f:
        return [line.strip() for line in f]


class Settings(BaseSettings):
    RTMP_URL: str
    OUTPUT_UDP_URL: str
    MODEL: str

    class Config:
        env_file = ".env"


settings = Settings()
exclude_phrases = get_exclude_phrases()
