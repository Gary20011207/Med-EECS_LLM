# Med-EECS_LLM

## Naive RAG Pipeline
### Installation
```
# I use CUDA 12.6 with RTX 4090.
conda create -n langchain-rag python=3.10 -y
conda activate langchain-rag
pip install -r requirements.txt
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
