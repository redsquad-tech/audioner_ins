import io
import json
import os
from typing import Optional

import openai
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from src.settings import PROMPTS, DIR2DIARISATION, BATCH_SIZE

torch.cuda.empty_cache()
api_base = os.environ.get("API_BASE")
api_key = os.environ.get("OPENAI_KEY")
model = os.environ.get("MODEL")


class NER:
    def __init__(
            self,
            chat_gpt_model_name: str = "gpt-4o",
            whisper_model_name: str = "openai/whisper-large-v3",
            language: Optional[str] = "russian",
    ):
        self.chat_gpt_model_name = chat_gpt_model_name
        self.whisper_model_name = whisper_model_name
        self.language = language

        # Initialize device and accuracy
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(self.device, self.torch_dtype)

        # Initialize openai client
        self.client = openai.OpenAI(
            base_url=api_base,
            api_key=api_key
        )

        print("Loading model...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.whisper_model_name, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(self.device)
        processor = AutoProcessor.from_pretrained(self.whisper_model_name)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            return_timestamps=True,
            batch_size=BATCH_SIZE,
            chunk_length_s=30,
            generate_kwargs={"language": self.language}
        )

    @staticmethod
    def clean_json(json_str: str, mode: str = "json") -> str:
        if json_str[0: len(f"```{mode}\n")] == f"```{mode}\n":
            json_str = json_str.removeprefix(f"```{mode}\n")
            json_str = json_str.removesuffix("```")
        return json_str

    def __call__(self, paths: str | io.BytesIO, keys: str) -> tuple[dict[str, str | None], dict[str, str | None]]:
        print("Computing")

        output = self.pipe(paths, batch_size=len(paths))
        jsons: dict[str, str | None] = {}
        transcriptions: dict[str, str | None] = {}

        for path, transcript_segments, key in zip(paths, output, keys):
            full_transcript = transcript_segments["text"]

            # Check and run ner
            if PROMPTS[key]:
                message_ner = PROMPTS[key].format(full_transcript)
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": message_ner}],
                    model=self.chat_gpt_model_name
                )
                json_text = json.loads(self.clean_json(chat_completion.choices[0].message.content))
            else:
                json_text = None

            # Check and run diarization
            if DIR2DIARISATION[key]:
                message_diarisation = PROMPTS["diarisation"].format(full_transcript)
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": message_diarisation}],
                    model=self.chat_gpt_model_name
                )
                diarisation_transcript = json.loads(self.clean_json(chat_completion.choices[0].message.content))
            else:
                diarisation_transcript = None

            jsons[path] = json_text
            transcriptions[path] = diarisation_transcript
        torch.cuda.empty_cache()
        return jsons, transcriptions


ner = NER(
    chat_gpt_model_name=model,
    whisper_model_name="openai/whisper-large-v3",
    language="russian",
)
