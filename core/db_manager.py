# core/db_manager.py
import logging
import os
import shutil # 用於刪除資料夾
from typing import List, Dict, Any, Optional, Tuple, Iterable

import chromadb
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 從 config.py 導入預設值
from config import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_VECTOR_STORE_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_K_SEARCH_RESULTS
)

logger = logging.getLogger(__name__)

class DBManager:
    """向量資料庫管理器 (ChromaDB)"""

    def __init__(self,
                 embedding_model_name: Optional[str] = None,
                 vector_store_path: Optional[str] = None,
                 collection_name: Optional[str] = None,
                 chunk_size: Optional[int] = None,
                 chunk_overlap: Optional[int] = None,
                 default_k_search: Optional[int] = None):
        """
        初始化 DBManager。

        Args:
            embedding_model_name: 用於生成嵌入的 SentenceTransformer 模型名稱。
            vector_store_path: ChromaDB 持久化儲存的路徑。
            collection_name: ChromaDB 中的集合名稱。
            chunk_size: 文本分割時的塊大小。
            chunk_overlap: 文本分割時塊之間的重疊大小。
            default_k_search: 執行搜尋時預設返回的文檔數量。
        """
        self.embedding_model_name: str = embedding_model_name or DEFAULT_EMBEDDING_MODEL_NAME
        self.vector_store_path: str = vector_store_path or DEFAULT_VECTOR_STORE_PATH
        self.collection_name: str = collection_name or DEFAULT_COLLECTION_NAME
        self.chunk_size: int = chunk_size if chunk_size is not None else DEFAULT_CHUNK_SIZE
        self.chunk_overlap: int = chunk_overlap if chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP
        self.default_k: int = default_k_search if default_k_search is not None else DEFAULT_K_SEARCH_RESULTS

        self.embedding_function: Optional[embedding_functions.SentenceTransformerEmbeddingFunction] = None
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[chromadb.Collection] = None
        self.is_connected: bool = False # 追蹤連線狀態

        logger.info(f"DBManager initialized: path='{self.vector_store_path}', collection='{self.collection_name}', "
                    f"embedding_model='{self.embedding_model_name}', chunk_size={self.chunk_size}, k={self.default_k}")

        self._get_embedding_model() # 初始化時即取得 embedding function

    def _get_embedding_model(self) -> None:
        """獲取 SentenceTransformer 嵌入模型函數。"""
        if self.embedding_function is None:
            try:
                logger.info(f"正在載入嵌入模型: {self.embedding_model_name}")
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_model_name
                )
                logger.info("嵌入模型載入成功。")
            except Exception as e:
                logger.error(f"載入嵌入模型 '{self.embedding_model_name}' 失敗: {e}", exc_info=True)
                # 根據需求，這裡可以引發異常或允許在沒有嵌入函數的情況下繼續 (但多數操作會失敗)
                raise  # 或者 return None，並在其他地方處理 self.embedding_function 為 None 的情況

    def connect(self, create_if_not_exists: bool = True) -> bool:
        """
        連接到 ChromaDB 並獲取/創建指定的集合。

        Args:
            create_if_not_exists: 如果集合不存在，是否創建它。

        Returns:
            True 如果成功連接並獲取/創建集合，否則 False。
        """
        if self.is_connected and self.client and self.collection:
            logger.debug(f"已連接到集合 '{self.collection_name}' at '{self.vector_store_path}'")
            return True

        if self.embedding_function is None:
            logger.error("無法連接資料庫：嵌入函數未初始化。")
            return False

        try:
            logger.info(f"嘗試連接到 ChromaDB，路徑: '{self.vector_store_path}'")
            # 每次 connect 都重新建立 client，以確保狀態最新，特別是在 reset/delete 後
            self.client = chromadb.PersistentClient(path=self.vector_store_path)
            logger.debug("ChromaDB PersistentClient 實例化成功。")

            try:
                logger.debug(f"嘗試獲取集合: '{self.collection_name}'")
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function # 傳遞實例而非僅名稱
                )
                logger.info(f"成功連接到現有集合: '{self.collection_name}'")
            except Exception as e_get: # 更通用的異常捕獲，因為底層錯誤可能不是 CollectionNotFound
                logger.warning(f"獲取集合 '{self.collection_name}' 失敗: {e_get}")
                if create_if_not_exists:
                    logger.info(f"集合 '{self.collection_name}' 不存在，嘗試創建...")
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        embedding_function=self.embedding_function, # 傳遞實例
                        metadata={"hnsw:space": "cosine"}  # L2 is default, cosine often better for ST
                    )
                    logger.info(f"成功創建新集合: '{self.collection_name}'")
                else:
                    logger.warning(f"未創建集合 '{self.collection_name}' (create_if_not_exists=False)。")
                    self.collection = None
                    self.is_connected = False
                    return False
            
            self.is_connected = True
            return True

        except Exception as e:
            logger.error(f"連接到 ChromaDB 或獲取/創建集合 '{self.collection_name}' 失敗: {e}", exc_info=True)
            self.client = None
            self.collection = None
            self.is_connected = False
            return False

    def _ensure_connected(self) -> None:
        """確保已連接到資料庫和集合。如果未連接，則嘗試連接。"""
        if not self.is_connected or not self.collection:
            logger.debug("_ensure_connected: 未連接，嘗試連接...")
            if not self.connect(create_if_not_exists=True): # 預設會嘗試創建
                raise ConnectionError(f"無法連接或初始化資料庫集合 '{self.collection_name}'。請檢查日誌。")


    def load_and_split_documents(self, file_paths: List[str]) -> List[Document]:
        """從文件路徑列表載入並分割文檔。"""
        if not file_paths:
            logger.warning("沒有提供文件路徑進行載入。")
            return []

        raw_documents = []
        for file_path in file_paths:
            try:
                # 這裡假設是 txt 文件，您可以擴展以支援 pdf, docx 等
                if not os.path.exists(file_path):
                    logger.warning(f"文件不存在，跳過: {file_path}")
                    continue
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 創建 Langchain Document，source 作為 metadata
                    # 檔名通常是好的 source 資訊
                    file_name = os.path.basename(file_path)
                    raw_documents.append(Document(page_content=content, metadata={"source_file_name": file_name}))
                logger.info(f"已載入文件: {file_path}")
            except Exception as e:
                logger.error(f"載入文件 {file_path} 失敗: {e}", exc_info=True)

        if not raw_documents:
            logger.warning("未能從提供的路徑載入任何文檔。")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True, # 添加塊在原文中的起始索引到 metadata
        )
        split_docs = text_splitter.split_documents(raw_documents)
        logger.info(f"已將 {len(raw_documents)} 個原始文檔分割成 {len(split_docs)} 個文本塊。")
        return split_docs

    def add_documents(self, documents: List[Document]) -> bool:
        """
        將 Langchain Document 列表添加到 ChromaDB 集合中。
        文檔應已包含 page_content 和 metadata。
        """
        self._ensure_connected()
        if not documents:
            logger.warning("沒有提供文檔進行添加。")
            return False
        if self.collection is None: # 額外檢查，雖然 _ensure_connected 應該處理
             logger.error("無法添加文檔：集合未初始化。")
             return False

        try:
            contents = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            # 創建唯一的 ID，例如使用 source 和 chunk index (如果可用)
            ids = []
            for i, doc in enumerate(documents):
                source = doc.metadata.get("source_file_name", "unknown_source")
                start_index = doc.metadata.get("start_index", i) # 使用 start_index 或序列號
                unique_id = f"{source}_chunk{start_index}"
                # 確保ID對ChromaDB有效 (例如，不能太長或包含非法字符，儘管Chroma通常處理得很好)
                ids.append(unique_id)


            logger.info(f"準備將 {len(documents)} 個文本塊添加到集合 '{self.collection_name}'...")
            # ChromaDB 的 add 方法會自動處理嵌入
            self.collection.add(
                documents=contents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"成功將 {len(documents)} 個文本塊添加到集合 '{self.collection_name}'。")
            return True
        except Exception as e:
            logger.error(f"添加文檔到集合 '{self.collection_name}' 失敗: {e}", exc_info=True)
            return False

    def search(self, query: str, k: Optional[int] = None, 
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        在向量資料庫中搜尋與查詢相關的文檔。

        Args:
            query: 查詢文本。
            k: 要返回的結果數量。如果為 None，則使用 self.default_k。
            filter_metadata: 用於過濾結果的元數據字典 (例如 {"source_file_name": "doc1.pdf"})。

        Returns:
            Langchain Document 對象列表。
        """
        self._ensure_connected()
        if self.collection is None:
             logger.error("無法搜尋：集合未初始化。")
             return []

        num_results = k if k is not None else self.default_k
        logger.debug(f"在集合 '{self.collection_name}' 中搜尋 '{query[:50]}...', k={num_results}, filter={filter_metadata}")

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=num_results,
                where=filter_metadata,  # ChromaDB 使用 'where' 進行元數據過濾
                include=['documents', 'metadatas', 'distances'] # 包含距離有助於調試
            )
            
            # 將 ChromaDB 結果轉換回 Langchain Document 格式
            langchain_docs = []
            if results and results.get('ids') and results['ids'][0]: # results['ids'] 是一個列表的列表
                for i in range(len(results['ids'][0])):
                    doc_content = results['documents'][0][i] if results['documents'] else ""
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else None
                    if distance is not None: # 將距離添加到元數據中，方便參考
                        metadata['distance'] = distance
                    langchain_docs.append(Document(page_content=doc_content, metadata=metadata))
            
            logger.info(f"搜尋 '{query[:50]}...' 找到 {len(langchain_docs)} 個結果。")
            return langchain_docs
        except Exception as e:
            logger.error(f"在集合 '{self.collection_name}' 中搜尋失敗: {e}", exc_info=True)
            return []

    def get_collection_status(self) -> Dict[str, Any]:
        """獲取當前集合的狀態。"""
        status = {
            "vector_store_path": self.vector_store_path,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model_name,
            "chunk_size_setting": self.chunk_size,
            "chunk_overlap_setting": self.chunk_overlap,
            "default_k_setting": self.default_k,
            "is_connected": self.is_connected,
            "client_exists": self.client is not None,
            "collection_exists_in_client": False,
            "item_count": 0,
        }
        if self.is_connected and self.collection:
            try:
                status["item_count"] = self.collection.count()
                status["collection_exists_in_client"] = True
            except Exception as e:
                logger.warning(f"獲取集合 '{self.collection_name}' 狀態時出錯: {e}")
                status["item_count"] = "Error"
        elif self.client: # Client存在但collection可能不存在或未連接
             try:
                # 嘗試獲取而不創建，來判斷是否存在
                self.client.get_collection(name=self.collection_name, embedding_function=self.embedding_function)
                status["collection_exists_in_client"] = True
                # 無法獲取 item_count，因為 self.collection 未設定
             except:
                status["collection_exists_in_client"] = False
        return status

    def delete_collection_data(self, confirm: bool = False) -> bool:
        """
        刪除指定集合中的所有數據。集合本身將保留，但為空。

        Args:
            confirm: 必須設為 True 以執行刪除操作。

        Returns:
            True 如果操作成功（或集合本不存在），False 如果操作失敗或未確認。
        """
        if not confirm:
            logger.warning("刪除集合數據操作未確認。請傳入 confirm=True 以繼續。")
            return False

        self._ensure_connected() # 確保客戶端和集合對象存在
        if not self.collection:
             logger.warning(f"無法刪除集合數據：集合 '{self.collection_name}' 未初始化或不存在。")
             return True # 從某種意義上說，它已經是空的了

        try:
            logger.warning(f"準備刪除集合 '{self.collection_name}' 中的所有數據...")
            # 獲取集合中所有 ID (ChromaDB 沒有直接的 clear 方法，需要刪除所有項目)
            # 一個更簡單的方式是刪除並重建集合，但題目要求的是"刪除資料"
            # 如果資料量非常大，get() 可能有性能問題。
            # ChromaDB 的 client.delete_collection() 然後 client.create_collection() 是更可靠的清空方式。
            # 或者，如果只想清空內容但保留集合結構 (這在Chroma中不直接支持，因為元數據等也與collection綁定)
            # 我們這裡採用刪除再創建同名 collection 的方式來"清空"它。
            collection_id = self.collection.id
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"舊集合 '{self.collection_name}' (ID: {collection_id}) 已刪除。")
            # 立刻重新創建同名集合
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"已重新創建空集合 '{self.collection_name}'。")
            return True
        except Exception as e:
            logger.error(f"刪除集合 '{self.collection_name}' 的數據失敗: {e}", exc_info=True)
            # 發生錯誤後，集合狀態可能不確定，最好重置
            self.collection = None
            self.is_connected = False
            return False

    def fully_delete_vector_store(self, confirm: bool = False) -> bool:
        """
        完全刪除向量儲存，包括其持久化路徑上的所有內容。
        此操作會將 DBManager 的 client 和 collection 設為 None。

        Args:
            confirm: 必須設為 True 以執行刪除操作。

        Returns:
            True 如果操作成功，False 如果操作失敗或未確認。
        """
        if not confirm:
            logger.warning("完全刪除向量儲存操作未確認。請傳入 confirm=True 以繼續。")
            return False

        logger.warning(f"準備完全刪除向量儲存於路徑: '{self.vector_store_path}'...")

        # 1. 嘗試使用 Chroma client 的 reset (如果 client 存在)
        if self.client:
            try:
                logger.info("正在重設 ChromaDB client...")
                self.client.reset() # 清除此客戶端路徑下的所有數據
                logger.info("ChromaDB client 重設完成。")
            except Exception as e:
                logger.error(f"ChromaDB client 重設失敗: {e}", exc_info=True)
                # 即使 reset 失敗，我們仍繼續嘗試刪除資料夾

        # 2. 將內部 client 和 collection 參考設為 None
        self.client = None
        self.collection = None
        self.is_connected = False
        logger.debug("內部 client 和 collection 參考已清除。")

        # 3. 刪除物理資料夾
        if os.path.exists(self.vector_store_path):
            try:
                shutil.rmtree(self.vector_store_path)
                logger.info(f"成功刪除向量儲存資料夾: '{self.vector_store_path}'")
                return True
            except OSError as e: # shutil.rmtree 可能因權限等問題失敗
                logger.error(f"刪除向量儲存資料夾 '{self.vector_store_path}' 失敗: {e}", exc_info=True)
                return False
        else:
            logger.info(f"向量儲存資料夾 '{self.vector_store_path}' 本不存在。")
            return True # 資料夾不存在也視為"已刪除"狀態

    def rebuild_database_from_files(self, file_paths: List[str], confirm_delete_existing: bool = False) -> bool:
        """
        從指定的文件列表完全重建向量資料庫。
        這將首先刪除現有的向量儲存（如果 confirm_delete_existing 為 True），然後重新連接並添加新文檔。

        Args:
            file_paths: 要載入和嵌入的文件路徑列表。
            confirm_delete_existing: 必須設為 True 才會刪除現有的向量儲存。

        Returns:
            True 如果重建成功，否則 False。
        """
        logger.info(f"開始從文件重建資料庫: path='{self.vector_store_path}', collection='{self.collection_name}'")

        if not self.fully_delete_vector_store(confirm=confirm_delete_existing):
            if confirm_delete_existing: # 如果用戶確認要刪但刪除失敗
                 logger.error("重建資料庫失敗：未能成功刪除現有向量儲存。")
                 return False
            # 如果用戶未確認刪除，則 proceed_without_delete 為 True，但我們仍需要一個乾淨的 client
            logger.info("未確認刪除現有儲存，或儲存本不存在。將嘗試在現有路徑上操作。")
            self.client = None # 確保 client 被重新實例化
            self.collection = None
            self.is_connected = False


        # 重新連接 (這將創建新的 PersistentClient 和集合)
        if not self.connect(create_if_not_exists=True):
            logger.error("重建資料庫失敗：未能連接或創建新集合。")
            return False

        # 載入和分割文檔
        logger.info("正在載入和分割新文檔...")
        documents_to_add = self.load_and_split_documents(file_paths)
        if not documents_to_add:
            logger.warning("沒有從提供的文件路徑中準備好任何文檔進行添加。資料庫可能為空。")
            return True # 可以認為是"成功"建立了空資料庫

        # 添加文檔
        logger.info("正在將文檔添加到新資料庫...")
        if self.add_documents(documents_to_add):
            logger.info("資料庫重建並成功填充數據。")
            return True
        else:
            logger.error("資料庫重建後填充數據失敗。")
            return False

    def disconnect(self) -> None:
        """
        斷開與 ChromaDB 的連接並釋放資源 (主要是將引用設為 None)。
        ChromaDB PersistentClient 通常不需要顯式的 close() 方法，
        但將引用設為 None 有助於垃圾回收和表明不再使用。
        """
        logger.info(f"正在斷開與 ChromaDB collection '{self.collection_name}' 的連接...")
        self.collection = None
        self.client = None # 釋放 client 引用
        self.is_connected = False
        logger.info("已斷開連接。")

    def __del__(self):
        """確保在對象銷毀時斷開連接。"""
        try:
            if self.is_connected:
                self.disconnect()
        except Exception as e:
            # 在 __del__ 中最好不要拋出異常
            try:
                logger.warning(f"DBManager __del__ 中斷開連接時發生錯誤: {e}")
            except: # logging 本身也可能失敗
                pass