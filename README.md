# Med-EECS_LLM

## Naive RAG Pipeline
https://a7b695a5096a2cffd5.gradio.live
## I use CUDA 12.6 with RTX 4090.
### Installation
```
conda create -n langchain-rag python=3.10 -y
conda activate langchain-rag
pip install -r RAG_requirements.txt
```
### Run
```
python3 RAG_NAIVE.py
```

## Web Server
### Installation
```
pip install -r requirements.txt
```
### Initialize Database
```
python3 init_db.py
```
### Run Server
```
python3 app.py
```
