# Docker

* 使用 CUDA 12.8 + PyTorch 環境

## 建立並啟動容器

根目錄執行：

```bash
docker compose -f docker/docker-compose.yml up --build -d
````

## 停止容器

```bash
docker compose -f docker/docker-compose.yml down
```


