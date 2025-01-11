import io
import os
from typing import Optional

import faster_whisper
import openai
import torch

torch.cuda.empty_cache()
openai.api_base = os.environ.get("API_BASE")


class NER:
    def __init__(
            self,
            chat_gpt_model_name: str = "gpt-4o",
            whisper_model_name: str = "large-v2",
            suppress_numerals: bool = True,
            language: Optional[str] = None,
            batch_size: int = 0
    ):
        self.chat_gpt_model_name = chat_gpt_model_name
        self.whisper_model_name = whisper_model_name
        self.suppress_numerals = suppress_numerals
        self.language = language
        self.batch_size = batch_size

        # Initialize device and accuracy
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "int8" if self.device == "cpu" else "float16"

        # Initialize openai client
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_KEY"))

        print("Loading model...")
        self.whisper_model = faster_whisper.WhisperModel(
            self.whisper_model_name, device=self.device, compute_type=self.compute_type
        )
        self.whisper_pipeline = faster_whisper.BatchedInferencePipeline(self.whisper_model)
        self.suppress_tokens = (
            self.find_numeral_symbol_tokens(self.whisper_model.hf_tokenizer)
            if self.suppress_numerals
            else [-1]
        )

    @staticmethod
    def find_numeral_symbol_tokens(tokenizer):
        numeral_symbol_tokens = [
            -1,
        ]
        for token, token_id in tokenizer.get_vocab().items():
            has_numeral_symbol = any(c in "0123456789%$£" for c in token)
            if has_numeral_symbol:
                numeral_symbol_tokens.append(token_id)
        return numeral_symbol_tokens

    @property
    def prompt(self) -> str:
        prompt = """
          Выдели следующие сущности из текста в формате json:
          ФИО застрахованного
          Дата рождения застрахованного
          Номер полиса 
          Диагноз/жалобы
          Услуги, которые необходимо согласовать
          Название клиники, в которую застрахованный планирует обратиться.
          Специализацию врача, к которому нужна запись
          Жалобы, в зависимости от специальности врача
          Название клиники, в которую застрахованный планирует обратиться.
        
          Текст - {}
        """
        return prompt

    def __call__(self, audio: str | io.BytesIO) -> dict:
        print("Computing")
        audio_waveform = faster_whisper.decode_audio(audio)

        if self.batch_size > 0:
            transcript_segments, info = self.whisper_pipeline.transcribe(
                audio_waveform,
                self.language,
                suppress_tokens=self.suppress_tokens,
                batch_size=self.batch_size,
                without_timestamps=True,
            )
        else:
            transcript_segments, info = self.whisper_model.transcribe(
                audio_waveform,
                self.language,
                suppress_tokens=self.suppress_tokens,
                without_timestamps=True,
                vad_filter=True,
            )

        full_transcript = "".join(segment.text for segment in transcript_segments)
        prompt = self.prompt.format(full_transcript)
        torch.cuda.empty_cache()

        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=self.chat_gpt_model_name,
        )

        json_text = chat_completion.choices[0].message.content
        if json_text[0: len("```json\n")] == "```json\n":
            json_text = json_text.removeprefix("```json\n")
            json_text = json_text.removesuffix("```")

        result = {
            "json": json_text,
            "transcription": full_transcript
        }
        return result


ner = NER(
    chat_gpt_model_name="gpt-4o",
    whisper_model_name="large-v2",
    suppress_numerals=True,
    language=None,
    batch_size=0
)
