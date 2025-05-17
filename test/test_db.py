# tests/test_db.py
"""
測試 DBManager 的功能，包括：
1. 指定集合名稱（作為客戶端ID）和檔案列表創建向量數據庫
2. 重新連接到數據庫
3. 重新建立數據庫
4. 切換不同的數據庫（通過更改集合名稱）
"""


# 導入 DBManager
from core.db_manager import DBManager


# 測試文件路徑
TEST_FILES_DIR = os.path.join(current_dir, 'test_files')
TEST_DB_DIR = os.path.join(current_dir, 'test_vector_db')

def prepare_test_files() -> List[str]:
    """準備測試文件並返回它們的路徑"""
    os.makedirs(TEST_FILES_DIR, exist_ok=True)
    
    # 創建一些測試文件
    file_paths = []
    for i in range(3):
        file_path = os.path.join(TEST_FILES_DIR, f'test_doc_{i}.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"這是測試文檔 {i}。\n" * 10)
            f.write(f"這裡包含一些關鍵字如：ERAS, 手術, 醫療, 復健, 營養。\n" * 3)
            f.write(f"文檔 {i} 的獨特內容。")
        file_paths.append(file_path)
    
    return file_paths

def cleanup():
    """清理測試文件和數據庫"""
    if os.path.exists(TEST_FILES_DIR):
        shutil.rmtree(TEST_FILES_DIR)
    if os.path.exists(TEST_DB_DIR):
        shutil.rmtree(TEST_DB_DIR)

def test_db_creation(client_id: str, file_paths: List[str]) -> DBManager:
    """
    測試 1: 通過指定 client_id 和文件列表創建向量數據庫
    這裡使用 collection_name 作為 client_id
    """
    logger.info(f"測試 1: 創建向量數據庫 - client_id: {client_id}")
    
    # 初始化 DBManager，使用 collection_name 作為 client_id
    db_manager = DBManager(
        vector_store_path=TEST_DB_DIR,
        collection_name=client_id,  # 使用 collection_name 作為 client_id
        chunk_size=500,             # 較小的值以便於測試
        chunk_overlap=50
    )
    
    # 連接到資料庫
    connected = db_manager.connect(create_if_not_exists=True)
    logger.info(f"連接到資料庫: {'成功' if connected else '失敗'}")
    
    # 從文件路徑中載入並分割文檔
    documents = db_manager.load_and_split_documents(file_paths)
    logger.info(f"載入並分割了 {len(documents)} 個文檔片段")
    
    # 添加文檔到資料庫
    added = db_manager.add_documents(documents)
    logger.info(f"添加文檔到資料庫: {'成功' if added else '失敗'}")
    
    # 獲取資料庫狀態
    status = db_manager.get_collection_status()
    logger.info(f"資料庫狀態: {status}")
    
    return db_manager

def test_reconnection(client_id: str) -> DBManager:
    """
    測試 2: 重新連接到現有資料庫
    """
    logger.info(f"測試 2: 重新連接到資料庫 - client_id: {client_id}")
    
    # 初始化新的 DBManager 實例，但使用相同的 collection_name (client_id)
    db_manager = DBManager(
        vector_store_path=TEST_DB_DIR,
        collection_name=client_id
    )
    
    # 連接到資料庫
    connected = db_manager.connect(create_if_not_exists=False)  # 不創建新的
    logger.info(f"重新連接到資料庫: {'成功' if connected else '失敗'}")
    
    # 獲取並顯示資料庫狀態
    status = db_manager.get_collection_status()
    logger.info(f"重連後資料庫狀態: {status}")
    
    # 進行簡單查詢測試
    query = "ERAS 營養"
    results = db_manager.search(query)
    logger.info(f"查詢 '{query}' 找到 {len(results)} 個結果")
    
    return db_manager

def test_rebuild_database(client_id: str, file_paths: List[str]) -> DBManager:
    """
    測試 3: 重建資料庫
    """
    logger.info(f"測試 3: 重建資料庫 - client_id: {client_id}")
    
    # 初始化 DBManager，使用相同的 collection_name
    db_manager = DBManager(
        vector_store_path=TEST_DB_DIR,
        collection_name=client_id
    )
    
    # 獲取重建前的資料庫狀態
    status_before = db_manager.get_collection_status()
    logger.info(f"重建前資料庫狀態: {status_before}")
    
    # 重建資料庫
    rebuilt = db_manager.rebuild_database_from_files(file_paths, confirm_delete_existing=True)
    logger.info(f"重建資料庫: {'成功' if rebuilt else '失敗'}")
    
    # 獲取重建後的資料庫狀態
    status_after = db_manager.get_collection_status()
    logger.info(f"重建後資料庫狀態: {status_after}")
    
    return db_manager

def test_switch_database(client_id1: str, client_id2: str) -> None:
    """
    測試 4: 切換資料庫（通過更改 collection_name）
    """
    logger.info(f"測試 4: 切換資料庫 - 從 {client_id1} 到 {client_id2}")
    
    # 連接到第一個資料庫
    db_manager = DBManager(
        vector_store_path=TEST_DB_DIR,
        collection_name=client_id1
    )
    connected1 = db_manager.connect(create_if_not_exists=False)
    status1 = db_manager.get_collection_status()
    logger.info(f"連接到資料庫 {client_id1}: {'成功' if connected1 else '失敗'}")
    logger.info(f"資料庫 {client_id1} 狀態: {status1}")
    
    # 記錄第一個資料庫中的項目數量
    count1 = status1['item_count']
    
    # 進行一次查詢
    query = "ERAS"
    results1 = db_manager.search(query)
    logger.info(f"在資料庫 {client_id1} 中查詢 '{query}' 找到 {len(results1)} 個結果")
    
    # 斷開連接
    db_manager.disconnect()
    logger.info(f"已斷開與資料庫 {client_id1} 的連接")
    
    # 切換到第二個資料庫
    db_manager = DBManager(
        vector_store_path=TEST_DB_DIR,
        collection_name=client_id2
    )
    connected2 = db_manager.connect(create_if_not_exists=False)
    status2 = db_manager.get_collection_status()
    logger.info(f"連接到資料庫 {client_id2}: {'成功' if connected2 else '失敗'}")
    logger.info(f"資料庫 {client_id2} 狀態: {status2}")
    
    # 記錄第二個資料庫中的項目數量
    count2 = status2['item_count']
    
    # 進行相同的查詢
    results2 = db_manager.search(query)
    logger.info(f"在資料庫 {client_id2} 中查詢 '{query}' 找到 {len(results2)} 個結果")
    
    # 驗證兩個資料庫是不同的
    logger.info(f"資料庫比較: {client_id1} 有 {count1} 個項目，{client_id2} 有 {count2} 個項目")
    logger.info(f"切換資料庫測試 {'成功' if count1 != count2 or len(results1) != len(results2) else '失敗 (可能是資料庫內容相似)'}")

def main():
    """執行所有測試"""
    try:
        # 準備測試文件
        file_paths = prepare_test_files()
        logger.info(f"準備了 {len(file_paths)} 個測試文件: {file_paths}")
        
        # 測試 1: 創建資料庫 - 客戶端 1
        client_id1 = "expert_1"
        db1 = test_db_creation(client_id1, file_paths)
        db1.disconnect()
        
        # 測試 2: 重新連接到現有資料庫
        db1_reconnected = test_reconnection(client_id1)
        db1_reconnected.disconnect()
        
        # 測試 3: 重建資料庫
        # 修改測試文件以便區分重建前後的內容
        modified_file_path = file_paths[0]
        with open(modified_file_path, 'a', encoding='utf-8') as f:
            f.write("\n這是重建後添加的新內容，包含關鍵詞：重建，測試。")
        
        db1_rebuilt = test_rebuild_database(client_id1, file_paths)
        db1_rebuilt.disconnect()
        
        # 測試 4: 創建另一個資料庫 - 客戶端 2
        client_id2 = "expert_2"
        # 修改一個文件，使兩個資料庫中的內容有所不同
        with open(file_paths[1], 'w', encoding='utf-8') as f:
            f.write("這是專屬於客戶端 2 的文檔。\n" * 5)
            f.write("包含特殊關鍵詞：客戶端2，專屬。\n" * 3)
        
        db2 = test_db_creation(client_id2, [file_paths[1]])  # 只使用一個修改過的文件
        db2.disconnect()
        
        # 測試 5: 切換資料庫
        test_switch_database(client_id1, client_id2)
        
        logger.info("所有測試完成")
    finally:
        # 清理測試文件和資料庫
        cleanup()
        logger.info("已清理測試環境")

if __name__ == "__main__":
    main()