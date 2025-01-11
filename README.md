## Device Resources

- `Only CPU mode` - 24 CPU/ 32GB RAM
- `CUDA mode` - 16 CPU /16GB RAM / GPU >= 16GB


## Environment

`python3 >= 3.10, <3.13` (`3.10` - base)

```shell
python3 -m venv venv   
. ./venv/bin/activate
pip install -r requirements.txt
```

## Run

```shell
dotenv -f .env set OPENAI_KEY <OPENAI_KEY>
dotenv -f .env run streamlit run streamlit.py
```

- http://localhost:8501/ - address web-ui
- Примеры аудио в `examples/`