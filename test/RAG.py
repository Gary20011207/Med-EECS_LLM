from core.rag_engine import RAGEngine
from core.db_manager import VectorDBManager
from core.model_manager import ModelManager

# 初始化組件
model_manager = ModelManager()
model, tokenizer, model_max_length = model_manager.initialize()

db_manager = VectorDBManager()
db = db_manager.connect_db()

rag = RAGEngine(model_manager, db_manager)  # 建議用小寫變量名

query = "ERAS 的核心目標是什麼？"

# 方法1: 使用 stream 生成
print("=== 流式生成回應 ===")
for text in rag.stream(query):
    print(text, end='', flush=True)

# 在流式生成完成後獲取資源
print("\n\n=== 資料來源 ===")
resources = rag.get_last_rag_resources()  # 正確的方法名
for resource in resources:
    print(f"- {resource['formatted_info']}")

# 方法2: 使用 generate 生成
print("\n=== 普通生成回應 ===")
response = rag.generate(query)
print(response)

# 同樣可以獲取資源
print("\n=== 資料來源 ===")
resources = rag.get_last_rag_resources()
for resource in resources:
    print(f"- {resource['formatted_info']}")
    
# 方法3: 單純獲取 RAG 信息（不生成回應）
print("\n=== 僅檢索資料 ===")
rag_info = rag.get_rag_info(query)
print(f"找到 {rag_info['total_found']} 個相關文檔")
print(f"上下文: {rag_info['context'][:100]}...")

# 也可以獲取詳細資源
resources = rag.get_last_rag_resources()
for resource in resources:
    print(f"- {resource['source_file_name']} | {resource['formatted_info']}")