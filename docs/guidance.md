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

### Run Web Server
```
python3 app.py
```
The database `chat.db` will be initialized the first time this is run.

### Default admin account
- Username: admin
- Password: nimda

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

