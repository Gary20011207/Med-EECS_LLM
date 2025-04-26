# ERAS 醫療專案管理系統

## 前言
隨著實證醫學的發展，現行醫療服務中的治療指引與臨床路徑（如 ERAS 術後加速復原計畫）日益完善。然而，這些指引往往過於敘述化，執行上高度依賴多團隊協作（營養師、護理、復健科、麻醉科、手術醫師等）。在人力不足或醫療資源有限的環境中，繁瑣的規章與分散的執行流程導致項目難以完整實現。目前，ERAS 計畫依賴專人管理和聯繫，難以實現自動化與整合。

## 目標
本專案利用大型語言模型（LLM）的強大文件整理與文字輸出能力，打造一個以 LLM 為核心的「ERAS 醫療專案個管師」。系統以對話方式引導病患完成衛教、追蹤待辦事項，並整合各醫療團隊的工作細項，實現自動化管理與高效協作。

## 開發項目

### 程式前台（基於 Web HTML）
1. **管理員系統**：
   - 建立病患清單與各醫療團隊清單。
   - 提供管理員介面以監控進度與分配任務。
2. **病患端登入系統**：
   - 病患登入後可查看並逐項完成個人化待辦清單（To-Do List）。

### 病患 To-Do List 頁面
1. **ERAS 指引引導**：
   - 利用 LLM 結合「ERAS 指引文本」，透過 RAG 或 Fine-tuning 技術提供初步計畫介紹與引導。
   - 以問答（QA）形式完成衛教，回答病患常見問題。
2. **廣泛性 QA 頁面**：
   - 提供全面的 ERAS 計畫介紹，涵蓋術前、術中、術後注意事項。
3. **LLM 與 RAG 方案優化**：
   - 評估並選擇最適合文本衛教的 LLM 模型與 RAG 實現方式。

### 病患影像 QA（次階段）
1. **視覺問答（VQA）功能**：
   - 利用具備 VQA 能力的 LLM，提供術後衛教並透過醫療圖像直觀解釋手術過程。
2. **影像解釋模型**：
   - 以術後影像（如 lumbar X-ray s/p screw fixation）及影像報告進行 VQA 模型 Fine-tuning，使 LLM 具備解釋病患影像的能力。

## 概念展示

### 前台
- **登入畫面**：提供管理員與病患的登入入口，介面簡潔且易用。
  
### 病患端
- **多頁面設計**：根據不同待辦事項，提供對應的頁面與功能。
- **LLM QA 頁面（標題：ERAS 介紹）**：
  ```
  Bot: 你好！我是你的 ERAS 個管師，將協助你了解整個計畫。
  Client: 我術前要空腹多久？
  Bot: 根據 ERAS 指引，術前需空腹 6 小時（固體食物），2 小時（清流質）。如有特殊情況，請與你的醫療團隊確認。
  ```
  - 提供建議提示問題，引導病患快速獲取資訊。

## 專案進度
- **總計畫時程**：8 週
- **目前進度**：
  - 架構設計與基礎文件已完成（參見 `/docs` 與 `/templates`）。
  - 初步 RAG 實現已於 `/apps/RAG_NAIVE.py` 中測試。
  - 前台模板（`/templates/login.html`, `/register.html`, `/chat.html`）已初步設計。

## 分工
- **前端開發**：負責 Web 介面設計與實現（`app.py`）。
- **後端開發**：負責 LLM 整合、RAG 系統（`RAG_NAIVE.py`）。
- **模型優化**：*LLM 與 VQA 模型的 Fine-tuning 與測試。*
- **文件管理**：維護會議記錄（`/docs/meeting_YYYY-MM-DD.md`）與指引文件（`/docs/guidance.md`）。

## 會議日期
- 2025-04-22：初步會議，討論專案架構與分工（請將計入存放於 `/docs/meeting_2025-04-25.md`）。
- 下次會議：2025-04-29，請參考最新會議記錄。

## 專案結構
```
/PDF
  ├── test.pdf                   # 測試用 PDF 文件
/docs
  ├── guidance.md               # 程式指引
  ├── meeting_2025-04-22.md     # 會議記錄
/templates
  ├── login.html                # 登入頁面模板
  ├── register.html             # 註冊頁面模板
  ├── chat.html                 # 聊天介面模板
/apps
  ├── RAG_NAIVE.py              # 初步 RAG 實現
  ├── RAG_requirements.txt      # LLM依賴
app.py                          # 主應用程式
requirements.txt                # 環境依賴
README.md                       # 專案說明
```

## 安裝與執行

### 1. 使用
#### 上傳文件
- **PDF 文件**：將 ERAS 相關的 PDF 文件（例如指引或衛教資料）複製到專案資料夾中的 `/PDF` 目錄。
  - 範例：將 `new_guideline.pdf` 放入 `/PDF` 資料夾，確保文件名無特殊字符。
- **Markdown 文件**：將會議記錄或指引文件（`.md` 格式）複製到 `/docs` 目錄。
  - 範例：將 `meeting_2025-05-01.md` 放入 `/docs`。
- **提交文件**：
  1. 透過 GitHub 網頁介面提交文件。
  2. 或將文件傳送至Discord，由成員代為上傳。

#### 查看 Prototype
- **網址**：
  - 主應用程式：訪問 [eras.haobba.org](https://eras.haobba.org) 查看 Web 介面。
  - RAG QA 測試：訪問 [erasllm.haobbc.org](https://erasllm.haobbc.org/) 體驗 LLM 問答功能。
- **注意**：目前應用部署於本地伺服器（規格：Intel i9-9900K, DDR4 64GB RAM, CUDA 12.8, NVIDIA RTX 4070s）

### 2. 程式開發
- **部署與系統需求**：請參閱 [`/docs/guidance.md`](./docs/guidance.md) 

## 聯繫
如有問題，請於 GitHub Issue 中提出。
