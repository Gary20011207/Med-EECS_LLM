# ERAS 醫療專案管理系統 - 病患導向智能個管師

## 專案宗旨

建立一個結合 RAG (檢索增強生成) 技術的 ERAS 智能個管師系統，為病患提供個人化的術後康復指導，提升 ERAS (Enhanced Recovery After Surgery) 計畫的執行效率和病患參與度。

## 系統定位

### 目標用戶
- **主要用戶**: 接受 ERAS 計畫的病患
- **次要用戶**: 醫護人員、管理員

### 解決問題
- 現行 ERAS 指引過於敘述化，病患理解困難
- 多團隊協作流程分散，缺乏統一管理平台
- 人力不足導致個人化指導服務受限
- 病患自主管理能力不足，依賴度過高

## 系統架構

### 前端系統架構

```
┌─────────────────┬─────────────────┬─────────────────┐
│   病患端介面    │   管理員介面    │   醫護端介面    │
├─────────────────┼─────────────────┼─────────────────┤
│ ▪ 個人 To-Do    │ ▪ 病患管理      │ ▪ 進度監控      │
│ ▪ ERAS QA Bot   │ ▪ 團隊管理      │ ▪ 任務分派      │
│ ▪ 衛教內容      │ ▪ 系統設定      │ ▪ 指引管理      │
│ ▪ 進度追蹤      │ ▪ 數據分析      │ ▪ 報告產生      │
└─────────────────┴─────────────────┴─────────────────┘
```
### 資料模型

```python
# 病患實體
class Patient:
    - patient_id: 病歷號
    - name: 姓名
    - surgery_type: 手術類型
    - surgery_date: 手術日期
    - eras_stage: 當前 ERAS 階段
    - todo_list: 個人化任務清單
    
# 任務實體
class Task:
    - task_id: 任務編號
    - patient_id: 病患編號
    - category: 任務分類 (術前/術後/復健等)
    - description: 任務描述
    - due_date: 截止日期
    - status: 完成狀態
    - assigned_team: 負責團隊
```

### 主要組件

```
┌─────────────────┬─────────────────┬─────────────────┐
│   Web Interface │   Core Engine   │   Data Storage  │
├─────────────────┼─────────────────┼─────────────────┤
│ Flask App       │ RAG Engine      │ Vector Database │
│ Templates       │ Model Manager   │ PDF Documents   │
│ Static Assets   │ DB Manager      │ ChromaDB        │
└─────────────────┴─────────────────┴─────────────────┘
```

### 技術棧

- **後端框架**: Flask
- **LLM 模型**: Qwen 系列 (可配置)
- **向量資料庫**: ChromaDB
- **文件處理**: LangChain + PyPDF
- **嵌入模型**: sentence-transformers/all-MiniLM-L6-v2
- **前端**: HTML/CSS/JavaScript

## 檔案結構說明

### 核心模組 (`core/`)

```python
core/
├── __init__.py          # 模組初始化
├── model_manager.py     # LLM 模型管理、GPU/CPU 調度
├── db_manager.py        # 向量資料庫管理、PDF 處理
└── rag_engine.py        # RAG 檢索增強生成引擎
```

### 主應用程式

```python
├── app_chat_v2.py       # Flask 主應用，API 路由
├── config.py            # 系統配置設定
```

### 前端模板 (`templates/`)

```
templates/
├── chat_test_v2.html    # 主要聊天介面 (最新版)
├── chat_test.html       # 舊版聊天介面
└── [其他 HTML 模板]     # 登入、註冊、管理等頁面
```

## 工作流程

### 1. 系統初始化

```
啟動 → 載入配置 → 初始化模型管理器 → 
       連接向量資料庫 → 初始化 RAG 引擎 → 
       啟動 Flask 服務
```

### 2. RAG 處理流程

```
用戶問題 → 向量檢索 → 相關文檔擷取 → 
          構建 Prompt → LLM 生成回答 → 
          後處理 → 返回結果
```

### 3. API 端點

- `GET /api/status`: 系統狀態查詢
- `POST /api/chat`: 一般問答請求
- `POST /api/chat/stream`: 串流問答請求
- `POST /api/db/rebuild`: 重建向量資料庫
- `GET /api/db/source-files`: 獲取可用源檔案

## 關鍵特性

### 模型管理
- 支援模型 GPU/CPU 自動調度
- 閒置時自動釋放 GPU 記憶體
- 支援 4-bit/8-bit 量化

### RAG 功能
- 支援指定特定文件進行檢索
- 可調整檢索相關度參數
- 支援對話歷史記憶

### 使用者介面
- 即時串流回覆
- 支援對話記憶開關
- RAG 檢索開關
- 模型參數調整 (Temperature, Max Tokens)

## 配置參數 (`config.py`)

```python
# 模型配置
LLM_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-1M"
QUANTIZATION_LEVEL = "4bit"

# 資料庫配置
PDF_FOLDER = "./PDFS"
DB_PATH = "./VectorDB"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 生成參數
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_NEW_TOKENS = 1000
```

## 開發指引

### 修改檢索邏輯
- 檔案位置: `core/db_manager.py`
- 關鍵方法: `search()`, `process_search_results()`

### 調整模型行為
- 檔案位置: `core/rag_engine.py`
- 關鍵方法: `generate()`, `stream()`, `_build_prompt()`

### 更新介面功能
- 檔案位置: `templates/chat_test_v2.html`
- 主要區域: JavaScript 事件處理, API 調用邏輯

### 新增 API 端點
- 檔案位置: `app_chat_v2.py`
- 參考現有路由結構添加新功能

## 常見調整場景

1. **更換模型**: 修改 `config.py` 中的 `LLM_MODEL_NAME`
2. **調整檢索策略**: 修改 `db_manager.py` 中的檢索參數
3. **客製化 Prompt**: 修改 `config.py` 中的 `SYSTEM_PROMPT`
4. **介面優化**: 修改 `chat_test_v2.html` 中的樣式和互動邏輯
5. **新增功能**: 在 `app_chat_v2.py` 中添加新的 API 端點

## 部署注意事項

- 確保 PDF 檔案放置在 `./PDFS` 目錄
- 首次啟動會自動建立向量資料庫
- 需要足夠的 GPU 記憶體以支援選定的模型
- 可透過環境變數覆蓋預設配置

## 故障排除

1. **模型載入失敗**: 檢查 GPU 記憶體是否足夠，可嘗試使用更小的模型
2. **PDF 無法處理**: 確認檔案格式正確且無損壞
3. **檢索結果為空**: 檢查向量資料庫是否正確建立
4. **回答品質不佳**: 調整 Temperature 參數或更新系統 Prompt