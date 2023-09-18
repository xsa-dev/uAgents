import os
import unittest
from unittest.mock import patch, MagicMock
from ..whisper.messages.basic.messages import *
from ..whisper.whisper.agents.wisper_agent import whisper_agent

API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v2"
HUGGING_FACE_ACCESS_TOKEN = os.getenv("HUGGING_FACE_ACCESS_TOKEN")
headers = {"Authorization": "Bearer {HUGGING_FACE_ACCESS_TOKEN}"}


class TestGetTranscription(unittest.TestCase):
    def test_hf_query_success(self, mock_open, mock_post):
        """
        Test case for successful transcription query.

        Args:
            mock_open: A mock object for the `open` function.
            mock_post: A mock object for the `post` function.
        """
        # Set up mock return values
        mock_open.return_value.__enter__.return_value.read.return_value = b"audio_data"
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = [{"transcription": "Hello"}]

        # Set up test data
        ctx = MagicMock()
        sender = "user"
        audio = "test_audio.ogg"

        # Call the function being tested
        whisper_agent.get_transcription(ctx, sender, audio)

        # Check if the expected function calls were made
        mock_open.assert_called_once_with(audio, "rb")
        mock_post.assert_called_once_with(API_URL, headers=headers, data=b"audio_data")
        ctx.send.assert_called_once_with(sender, messages.UAResponse(response={"transcription": "Hello"}))


    def test_hf_query_error(self, mock_open, mock_post):
        # Test case for error in transcription query
        mock_open.return_value.__enter__.return_value.read.return_value = b"audio_data"
        mock_post.return_value.status_code = 500
        mock_post.return_value.json.return_value = {"error": "Transcription failed"}

        ctx = MagicMock()
        sender = "user"
        audio = "test_audio.ogg"

        get_transcription(ctx, sender, audio)

        mock_open.assert_called_once_with(audio, "rb")
        mock_post.assert_called_once_with(API_URL, headers=headers, data=b"audio_data")
        ctx.send.assert_called_once_with(sender, Error(error="Error: Transcription failed"))


    def test_hf_query_exception(self, mock_open, mock_post):
        # Test case for exception in transcription query
        mock_open.return_value.__enter__.return_value.read.side_effect = Exception("File not found")

        ctx = MagicMock()
        sender = "user"
        audio = "audio_file"

        get_transcription(ctx, sender, audio)

        mock_open.assert_called_once_with(audio, "rb")
        mock_post.assert_not_called()
        ctx.send.assert_called_once_with(sender, Error(error="An exception occurred while processing the request: File not found"))


if __name__ == '__main__':
    unittest.main()