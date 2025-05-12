# ERAS 醫療專案 - 優化後的架構

## 目錄結構

```
eras_medical_system/
├── app.py                    # Flask 應用主檔案
├── config.py                 # 統一配置管理
├── core/                     # 核心功能模組
│   ├── __init__.py           # 模組初始化
│   ├── model_manager.py      # LLM 模型管理
│   ├── vector_db_manager.py  # 向量資料庫管理
│   └── rag_engine.py         # RAG 問答引擎
├── utils/                    # 工具函數
│   ├── __init__.py           # 模組初始化
│   ├── pdf_processor.py      # PDF 處理工具
│   ├── text_chunker.py       # 文本分塊工具
│   └── logging_setup.py      # 日誌設定
├── api/                      # API 路由
│   ├── __init__.py           # 模組初始化
│   ├── chat.py               # 聊天相關 API
│   ├── database.py           # 資料庫相關 API
│   └── status.py             # 狀態監控 API
├── static/                   # 靜態檔案
│   └── style.css             # CSS 樣式
├── templates/                # HTML 模板
│   └── chat.html             # 聊天介面
├── data/                     # 資料目錄
│   ├── pdfs/                 # PDF 文件
│   └── vectordb/             # 向量資料庫
├── tests/                    # 測試代碼
│   ├── test_model.py         # 模型測試
│   ├── test_vectordb.py      # 向量資料庫測試
│   └── test_rag.py           # RAG 引擎測試
├── requirements.txt          # 依賴套件
└── README.md                 # 專案說明
```

## 核心模組說明

### 1. `config.py` - 統一配置管理

集中管理所有配置參數，包括模型設定、資料庫路徑、分塊參數等。各模組從這裡導入配置，避免重複定義。

### 2. `core/model_manager.py` - LLM 模型管理

負責 LLM 模型的載入、卸載、設備管理和資源監控。提供統一介面獲取模型和分詞器，處理 CPU/GPU 轉移邏輯。

### 3. `core/vector_db_manager.py` - 向量資料庫管理

負責向量資料庫的建立、更新和查詢。包含智能更新機制，只在必要時重建資料庫。提供緩存和預處理功能。

### 4. `core/rag_engine.py` - RAG 問答引擎

實現 RAG 檢索、問答生成和回覆串流邏輯。包含改進的上下文過濾、相關性評分和提示詞構建策略。

### 5. `utils/` - 工具函數

提供各種輔助功能，包括 PDF 處理、文本分塊和日誌設定。模組化設計使得功能易於擴展和複用。

### 6. `api/` - API 路由

將 API 路由分離到單獨的模組，每個 API 類型（聊天、資料庫、狀態）都有專門的檔案處理，提高可維護性。

