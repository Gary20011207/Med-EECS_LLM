# 程式開發

## Git
```bash
git clone https://github.com/Gary20011207/Med-EECS_LLM.git
cd Med-EECS_LLM
```
## Conda env
```
conda create -n langchain-rag python=3.10 -y
conda activate langchain-rag
pip install -r requirements.txt
pip install -r ./apps/RAG_requirements.txt
```

## Web Entry

### Initialize Database
```
python3 init_db.py
```
### Run Web
```
python3 app.py
```

## Naive RAG Pipeline

### Run
```
python3 ./apps/RAG_NAIVE.py
```

## Deployment

### Run
```
nohup python3 app.py > ./app_log.txt 2>&1 &
nohup python3 ./apps/RAG_NAIVE.py > ./rag_log.txt 2>&1 &
```

### Shutdown
```
kill 12345(PID)
```

