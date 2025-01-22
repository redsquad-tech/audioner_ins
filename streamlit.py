import shutil
import time
import zipfile
from pathlib import Path
import json

import streamlit as st
from src import ner
from src.settings import BATCH_SIZE


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield [t[0] for t in l[i:i + n]], [t[1] for t in l[i:i + n]]


def get_name_folder() -> str:
    name = time.strftime("%Y%m%d_%H%M%S")
    return name


def save_json(json_obj: dict, audio_filename: str, suffix: str) -> Path:
    path = Path(audio_filename)
    path = path.parent / f"{path.name}.{suffix}.json"
    with path.open("w") as json_file:
        json.dump(json_obj, json_file, ensure_ascii=False)
    return path


st.title("Загрузка архива")
uploaded_file = st.file_uploader("Загрузите архив с аудиофайлами", type=["zip"])

bar = st.empty()
load_button = st.empty()
metric = st.empty()
json1 = st.empty()
json2 = st.empty()

if uploaded_file is not None:
    root = Path(get_name_folder())
    path_jsons = []

    if root.exists():
        shutil.rmtree(root)
    root.mkdir(exist_ok=True)

    with zipfile.ZipFile(uploaded_file, 'r', metadata_encoding="utf-8") as zip_ref:
        zip_ref.extractall(root)

    dirs = [f for f in root.glob("*") if f.is_dir() and f.name[0] != "_"]
    audio_files = []
    for dir_ in dirs:
        audio_files += [
            (str(i), dir_.name)
            for i in dir_.glob("*")
            if i.is_file() and i.name[0] != "_" and i.name[0] != "."
        ]

    progress = bar.progress(0)
    for i, (audio_file, keys) in enumerate(list(divide_chunks(audio_files, BATCH_SIZE))):
        start_time = time.time()
        jsons, transcriptions = ner(audio_file, keys)
        end_time = time.time()
        processing_time = end_time - start_time

        progress.progress((i*BATCH_SIZE + min(BATCH_SIZE, len(audio_file))) / len(audio_files))
        metric.metric(f"Batch", f"{round(processing_time, 4)} sec", )
        json1.json(jsons)
        json2.json(transcriptions)

        for audio_path in audio_file:
            if jsons[audio_path] is not None:
                path_jsons.append(save_json(jsons[audio_path], audio_path, "ner"))
            if transcriptions[audio_path] is not None:
                path_jsons.append(save_json(transcriptions[audio_path], audio_path, "asr"))

    # Zip jsons ner/asr and load zip-archive
    result_zip_path = root / f"audioner_ins_{root.name}.zip"
    with zipfile.ZipFile(result_zip_path, mode='w') as zip_ref:
        for path in path_jsons:
            zip_ref.write(path)

    with result_zip_path.open("rb") as zip_file:
        load_button.download_button(
            label="Download zip-archive",
            data=zip_file,
            file_name=result_zip_path.name,
        )