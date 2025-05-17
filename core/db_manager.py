# simplified_db_manager.py
import os
import logging
import re
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# 導入預設值
from config import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_VECTOR_STORE_PATH,
    DEFAULT_DOCUMENTS_PATH, 
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_K_SEARCH_RESULTS
)

logger = logging.getLogger(__name__)

class DBManager:
    """簡化版向量資料庫管理器 (ChromaDB)"""

    def __init__(self,
                 embedding_model_name: Optional[str] = None,
                 vector_store_path: Optional[str] = None,
                 documents_path: Optional[str] = None,  # 新增：文檔路徑
                 chunk_size: Optional[int] = None,
                 chunk_overlap: Optional[int] = None,
                 default_k_search: Optional[int] = None):
        """
        初始化 SimplifiedDBManager

        Args:
            embedding_model_name: 嵌入模型名稱（也用作 collection 名稱）
            vector_store_path: ChromaDB 儲存路徑
            documents_path: PDF文檔資料夾路徑
            chunk_size: 文本分割大小
            chunk_overlap: 文本分割重疊大小
            default_k_search: 預設搜索結果數量
        """
        self.embedding_model_name = embedding_model_name or DEFAULT_EMBEDDING_MODEL_NAME
        self.vector_store_path = vector_store_path or DEFAULT_VECTOR_STORE_PATH
        self.documents_path = documents_path or DEFAULT_DOCUMENTS_PATH
        self.chunk_size = chunk_size if chunk_size is not None else DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP
        self.default_k = default_k_search if default_k_search is not None else DEFAULT_K_SEARCH_RESULTS
        
        # 將嵌入模型名稱轉換為有效的集合名稱
        self.collection_name = self._sanitize_collection_name(self.embedding_model_name)

        self.embedding_function = None
        self.client = None
        self.collection = None
        self.is_connected = False
        
        logger.info(f"SimplifiedDBManager 初始化: 嵌入模型='{self.embedding_model_name}', "
                   f"向量儲存路徑='{self.vector_store_path}', 文檔路徑='{self.documents_path}'") # 更新日誌
        logger.info(f"使用集合名稱: '{self.collection_name}'")
        
        # 初始化嵌入模型
        self._init_embedding_model()
    
    def _sanitize_collection_name(self, name: str) -> str:
        """
        將嵌入模型名稱轉換為有效的 ChromaDB 集合名稱
        
        ChromaDB 集合名稱要求:
        1. 3-63個字符
        2. 以字母數字開頭和結尾
        3. 只包含字母、數字、下劃線和連字符
        4. 不包含連續的點
        5. 不是有效的 IPv4 地址
        
        Args:
            name: 原始名稱
            
        Returns:
            str: 有效的集合名稱
        """
        # 替換無效字符為下劃線
        sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
        
        # 確保開頭和結尾是字母數字
        if not sanitized[0].isalnum():
            sanitized = 'model_' + sanitized
        if not sanitized[-1].isalnum():
            sanitized = sanitized + '_db'
            
        # 確保長度在有效範圍內
        if len(sanitized) < 3:
            sanitized = 'db_' + sanitized
        if len(sanitized) > 63:
            sanitized = sanitized[:60] + '_db'
            
        return sanitized
    
    def _init_embedding_model(self) -> None:
        """初始化嵌入模型"""
        try:
            logger.info(f"載入嵌入模型: {self.embedding_model_name}")
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )
            logger.info("嵌入模型載入成功")
        except Exception as e:
            logger.error(f"載入嵌入模型失敗: {e}", exc_info=True)
            raise
    
    def connect_db(self) -> bool:
        """
        連接到向量資料庫，如果對應 collection 不存在則創建
        
        Returns:
            bool: 連接是否成功
        """
        if self.is_connected and self.collection:
            logger.debug(f"已連接到集合 '{self.collection_name}'")
            return True
            
        try:
            # 創建或連接到 ChromaDB
            logger.info(f"連接到 ChromaDB: {self.vector_store_path}")
            self.client = chromadb.PersistentClient(path=self.vector_store_path)
            
            # 嘗試獲取現有集合
            try:
                logger.debug(f"嘗試獲取集合: {self.collection_name}")
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"已連接到現有集合: {self.collection_name}")
                self.is_connected = True
                return True
            except Exception as e:
                logger.warning(f"集合 '{self.collection_name}' 不存在，將創建新集合")
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"已創建新集合: {self.collection_name}")
                self.is_connected = True
                return True
                
        except Exception as e:
            logger.error(f"連接資料庫失敗: {e}", exc_info=True)
            self.client = None
            self.collection = None
            self.is_connected = False
            return False
    
    def _ensure_connected(self) -> None:
        """確保已連接到資料庫"""
        if not self.is_connected or not self.collection:
            if not self.connect_db():
                raise ConnectionError(f"無法連接到資料庫 '{self.collection_name}'")
    
    def rebuild_db(self, pdf_folder_path: Optional[str] = None) -> bool:
        """
        重建資料庫：刪除對應 collection 並重新創建
        
        Args:
            pdf_folder_path: PDF 文件資料夾路徑，如果為 None 則使用 self.documents_path
            
        Returns:
            bool: 是否成功重建
        """
        pdf_folder = pdf_folder_path or self.documents_path # 使用提供的路徑或默認文檔路徑
        
        # 檢查 PDF 文件夾是否存在
        if not os.path.exists(pdf_folder):
            logger.error(f"PDF 文件夾不存在: {pdf_folder}")
            return False
            
        # 獲取 PDF 文件列表
        pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) 
                     if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"在 {pdf_folder} 中沒有找到 PDF 文件")
            return False
            
        logger.info(f"找到 {len(pdf_files)} 個 PDF 文件")
        
        # 連接到資料庫
        try:
            if self.client is None:
                self.client = chromadb.PersistentClient(path=self.vector_store_path)
                
            # 嘗試刪除現有集合
            try:
                logger.info(f"刪除現有集合: {self.collection_name}")
                self.client.delete_collection(name=self.collection_name)
                self.collection = None
                self.is_connected = False
            except Exception as e:
                logger.warning(f"刪除集合失敗 (可能不存在): {e}")
            
            # 創建新集合
            logger.info(f"創建新集合: {self.collection_name}")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            self.is_connected = True
            
            # 處理 PDF 文件並添加到數據庫
            return self._process_and_add_pdfs(pdf_files)
            
        except Exception as e:
            logger.error(f"重建數據庫失敗: {e}", exc_info=True)
            return False
    
    def _process_and_add_pdfs(self, pdf_files: List[str]) -> bool:
        """
        處理 PDF 文件並添加到數據庫
        
        Args:
            pdf_files: PDF 文件路徑列表
            
        Returns:
            bool: 是否成功處理並添加
        """
        try:
            documents = []
            
            # 載入並分割所有 PDF
            for pdf_path in pdf_files:
                try:
                    logger.info(f"處理 PDF: {pdf_path}")
                    loader = PyPDFLoader(pdf_path)
                    pdf_docs = loader.load()
                    
                    # 為每個文檔添加源文件名
                    file_name = os.path.basename(pdf_path)
                    for doc in pdf_docs:
                        doc.metadata["source_file_name"] = file_name
                    
                    documents.extend(pdf_docs)
                except Exception as e:
                    logger.error(f"處理 PDF {pdf_path} 失敗: {e}")
                    continue
            
            if not documents:
                logger.warning("沒有成功處理任何 PDF 文件")
                return False
                
            # 分割文檔
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                add_start_index=True
            )
            
            chunks = text_splitter.split_documents(documents)
            logger.info(f"將 {len(documents)} 個文檔分割為 {len(chunks)} 個文本塊")
            
            # 添加到 ChromaDB
            contents = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            ids = []
            
            # 創建唯一 ID
            for i, chunk in enumerate(chunks):
                source = chunk.metadata.get("source_file_name", "unknown")
                page = chunk.metadata.get("page", 0)
                start_index = chunk.metadata.get("start_index", i)
                unique_id = f"{source}_p{page}_i{start_index}"
                ids.append(unique_id)
            
            # 添加到集合
            self.collection.add(
                documents=contents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"成功添加 {len(chunks)} 個文本塊到數據庫")
            return True
            
        except Exception as e:
            logger.error(f"處理和添加 PDF 失敗: {e}", exc_info=True)
            return False
    
    def search(self, query: str, k: Optional[int] = None, 
               source_files: Optional[List[str]] = None) -> List[Document]:
        """
        搜索相關文檔
        
        Args:
            query: 查詢文本
            k: 返回結果數量，若為 None 則使用默認值
            source_files: 源文件名列表進行過濾，若為 None 則搜索所有文件
            
        Returns:
            List[Document]: 包含相關文檔的列表
        """
        self._ensure_connected()
        
        results_count = k if k is not None else self.default_k
        filter_dict = {}
        
        # 如果指定了源文件進行過濾
        if source_files and len(source_files) > 0:
            filter_dict = {"source_file_name": {"$in": source_files}}
            
        try:
            logger.debug(f"搜索: '{query[:50]}...' (k={results_count}, 過濾={filter_dict})")
            
            # 執行搜索
            results = self.collection.query(
                query_texts=[query],
                n_results=results_count,
                where=filter_dict if filter_dict else None,
                include=['documents', 'metadatas', 'distances']
            )
            
            # 轉換為 Document 對象
            documents = []
            if results and results.get('ids') and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    content = results['documents'][0][i] if results['documents'] else ""
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else None
                    
                    if distance is not None:
                        metadata['distance'] = distance
                        
                    documents.append(Document(page_content=content, metadata=metadata))
            
            logger.info(f"搜索 '{query[:50]}...' 找到 {len(documents)} 個結果")
            return documents
            
        except Exception as e:
            logger.error(f"搜索失敗: {e}", exc_info=True)
            return []
    
    def get_available_source_files(self) -> List[str]:
        """
        獲取集合中可用的源文件清單
        
        Returns:
            List[str]: 源文件名列表
        """
        self._ensure_connected()
        
        try:
            # 使用 ChromaDB 的 get 方法獲取所有文檔
            all_docs = self.collection.get(include=['metadatas'])
            
            # 從元數據中提取唯一的源文件名
            source_files = set()
            if all_docs and 'metadatas' in all_docs:
                for metadata in all_docs['metadatas']:
                    if metadata and 'source_file_name' in metadata:
                        source_files.add(metadata['source_file_name'])
            
            file_list = sorted(list(source_files))
            logger.info(f"找到 {len(file_list)} 個源文件: {file_list}")
            return file_list
            
        except Exception as e:
            logger.error(f"獲取可用源文件失敗: {e}", exc_info=True)
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """
        獲取數據庫狀態
        
        Returns:
            Dict[str, Any]: 包含狀態信息的字典
        """
        status = {
            "embedding_model": self.embedding_model_name,
            "collection_name": self.collection_name,
            "vector_store_path": self.vector_store_path,
            "documents_path": self.documents_path,  # 添加文檔路徑到狀態信息
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "is_connected": self.is_connected,
            "doc_count": 0
        }
        
        if self.is_connected and self.collection:
            try:
                status["doc_count"] = self.collection.count()
                status["source_files"] = self.get_available_source_files()
            except Exception as e:
                logger.warning(f"獲取集合信息失敗: {e}")
                status["error"] = str(e)
        
        return status
    
    def disconnect(self) -> None:
        """斷開連接並釋放資源"""
        logger.info("斷開數據庫連接")
        self.collection = None
        self.client = None
        self.is_connected = False