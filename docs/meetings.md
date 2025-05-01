# 書面報告
- 書面報告於 Jun 7 (Sat) 11:59pm 前上傳 NTU COOL
    - 報告需要描述組員貢獻
    - 建議每組 4--8 頁 A4 （報告品質重於報告長度）
    - 字體大小以 12 號為主
    - 行間不另留間距
- 評分標準
    - 創意
    - 可行性與影響力
    - 技術深度
    - 完成度與完整性
    - 合作方式

---
# 2025/0429 Meeting
前台設計：
1. 醫師
2. 麻醉
3. 

---
# 2025/0422 Meeting
> [!Important]
> 開會跟老師約每週二 19:00
> 群組訊息請記得要回
1. 架Chat bot前台
    - 金哲安 （網頁）
    - 陳孟潔
    - 陳冠宇 (Phi-4 RAG Pipeline)
2. 搜集ERAS guidelines資料
    - 張玠
    - 楊哲瑜
3. Survey
    - 倪昕
    - 傅冠豪

---
# ERAS Notes 2025/0419
[ERAS for lumbar spinal fusion](https://www.sciencedirect.com/science/article/pii/S2666548424002737#tbl0001)
## 痛點
1. ERAS guidelines 太籠統
2. 個管師沒辦法管每個病人
## ERAS guidelines 太籠統（術前衛教）
### 方法
1. 利用 LLM 生成個人化衛教資訊，需要搜集病例與範例衛教說明，可以 Fine Tune LLM 或是利用 prompt engineering，prompt engineering 比較簡單
2. 利用 LLM 生成個人化衛教提醒清單，一樣使用 prompt engineering 比較簡單
### 用戶端 App 整合
1. 衛教資訊可以單純顯示在 App 中，或是跳出提醒通知
2. 提醒清單可以在 App 中列出，可以勾選，也可以加上提醒通知

## 個管師沒辦法管每個病人
### 方法
1. 利用 LLM 生成復健指導，需要預先寫好病例與對應的復健指導範例
2. 利用 LLM, VQA 回答病人疑問，需要 VQA 訓練資料
### 用戶端 App 整和
1. App 顯示復健指導，建立復健排程與通知提醒
2. App 回覆病人手術相關影像問題
3. App 收集滿意度調查
4. 回傳各項數據到醫生端

---
# 2025/0415 Meeting
陳孟潔 - 資工三
遊戲

倪昕 - 生工所 碩一
生態 運動

楊哲瑜 - 生工所 碩一
毒理
買藥 app

陳冠宇 - 資料科學 碩一 （資工）
LLM fine-tune

傅冠豪 - 精準健康 博一 （神經外科）（長庚）
Chat with （3D）image
術後復健菜單

張玠 - 醫學二
病歷
