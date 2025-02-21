import gc

import numpy as np
import torch
import whisperx

from src.services import logger


class SRTModel:
    """
    SRTModel provides functionality for audio transcription and diarization using the WhisperX library.
    """

    def __init__(
            self,
            model: str = 'large-v3-turbo',
            device: str = 'cuda',
            compute_type: str = 'float16',
            download_root: str = 'models',
    ):

        """
        Initialize the SRTModel with device, compute type, download root, and Hugging Face auth token.

        :param device: The device to run the model on (e.g., 'cuda' or 'cpu').
        :param compute_type: The type of computation to use (e.g., 'float16' or 'float32').
        :param download_root: The root directory to download the models to.
        """

        self._cache = {}
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self._load_model()

        logger.info(
            "SRTModel initialized with device=%s, compute_type=%s, download_root=%s",
            device,
            compute_type,
            download_root
        )

    def _load_model(self):

        """
        Load a specific transcription model if not already cached.

        :return: The loaded model.
        """

        if not self._cache.get(self.model):

            logger.info("Loading model %s...", self.model)

            try:
                self._cache[self.model] = whisperx.load_model(
                    self.model,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=self.download_root,
                )

                logger.info("Loaded model %s", self.model)

            except Exception as e:
                logger.error("Failed to load model %s: %s", self.model, e)
                raise e

        return self._cache[self.model]

    def transcribe(
            self,
            audio: np.ndarray,
            batch_size: int = 4,
            chunk_size: int = 10,
            language: str = None,
    ):
        """
        Transcribe an audio file and optionally perform diarization.

        :param audio: A NumPy array containing the audio waveform, in float32 dtype.
        :param batch_size: The batch size for transcription.
        :param chunk_size: The chunk size for transcription.
        :param language: The language of the audio.

        :return: The transcribed audio with speakers assigned.
        """

        model = self._load_model()

        try:
            logger.info("Transcribing audio...")

            result = model.transcribe(
                audio=audio,
                batch_size=batch_size,
                chunk_size=chunk_size,
                language=language,
            )

            logger.info("Transcribed audio file")

        except Exception as e:
            logger.error("Failed to transcribe audio file: %s", e)
            raise e

        return result['segments']

    def clean(self):

        """
        Clear the model cache and remove the diarization model.

        :return: None
        """

        logger.info("Cleaning up resources...")

        self._cache.clear()
        logger.info("Cleared model cache")

        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Collected garbage and cleared CUDA cache")

        logger.info("Cleanup complete")
