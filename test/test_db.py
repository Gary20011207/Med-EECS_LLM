# tests/test_db.py
from core.db_manager import VectorDBManager

db_manager = VectorDBManager()
db = db_manager.connect_db()

if db is not None:
    print("✅ 資料庫連接成功")
    source_files = db_manager.get_available_source_files()
    print(f"📄 資料庫中共收錄 {len(source_files)} 個 PDF 檔案")
else:
    print("❌ 無法連接或建立向量資料庫")

result = db_manager.search("ERAS")
print(result[0].page_content)