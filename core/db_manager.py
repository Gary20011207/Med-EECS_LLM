# core/db_manager.py
import os
import shutil
import logging
from typing import List, Optional, Tuple
import threading
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 導入配置
try:
    from config import (
        EMBEDDINGS_MODEL_NAME,
        PDF_FOLDER,
        DB_PATH,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        RAG_TOP_K
    )
except ImportError:
    # 如果無法導入配置，使用預設值
    EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    PDF_FOLDER = "./PDFS"
    DB_PATH = "./VectorDB"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RAG_TOP_K = 5

logger = logging.getLogger(__name__)

class VectorDBManager:
    """向量資料庫管理器"""
    
    def __init__(self):
        """初始化向量資料庫管理器"""
        self.pdf_folder = PDF_FOLDER
        self.db_path = DB_PATH
        self.embedding_model_name = EMBEDDINGS_MODEL_NAME
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
        self.default_top_k = RAG_TOP_K
        
        self.db = None
        self.embedding_function = None
        self._lock = threading.Lock()
        
        # 重建狀態追蹤
        self._rebuilding = False
        self._rebuild_status = {
            "status": "idle",
            "progress": 0,
            "message": "",
            "start_time": None,
            "total_files": 0,
            "processed_files": 0
        }
        
        logger.info(f"VectorDBManager 初始化完成")
        logger.info(f"  PDF 資料夾: {self.pdf_folder}")
        logger.info(f"  資料庫路徑: {self.db_path}")
        logger.info(f"  嵌入模型: {self.embedding_model_name}")
        logger.info(f"  預設 Top-K: {self.default_top_k}")
    
    def reset_and_rebuild_db(self,
                           pdf_folder: Optional[str] = None,
                           db_path: Optional[str] = None,
                           emb_model: Optional[str] = None,
                           chunk_size: Optional[int] = None,
                           chunk_overlap: Optional[int] = None,
                           force_reset: bool = True) -> Optional[Chroma]:
        """重建向量資料庫 - 處理所有 PDF 檔案
        
        Args:
            pdf_folder: PDF 檔案資料夾路徑
            db_path: 資料庫路徑
            emb_model: 嵌入模型名稱
            chunk_size: 文本塊大小
            chunk_overlap: 文本塊重疊
            force_reset: 是否強制重置資料庫
            
        Returns:
            Chroma 向量資料庫實例
        """
        with self._lock:
            if self._rebuilding:
                logger.warning("資料庫重建正在進行中，請等待")
                return None
            self._rebuilding = True
            
        # 使用參數或預設值
        pdf_folder = pdf_folder or self.pdf_folder
        db_path = db_path or self.db_path
        emb_model = emb_model or self.embedding_model_name
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        
        try:
            self._update_status("started", 0, "開始重建向量資料庫")
            logger.info(f"開始向量資料庫處理：PDF來源='{pdf_folder}', DB路徑='{db_path}', 模型='{emb_model}'")
            logger.info("將處理資料夾中的所有 PDF 檔案")
            
            # 1. 處理資料庫重置和目錄創建
            if force_reset and os.path.exists(db_path):
                logger.info(f"強制重置模式：正在刪除現有向量資料庫於 {db_path}")
                try:
                    # 先確保目錄有寫入權限
                    if os.path.exists(db_path):
                        # 遞歸設置目錄和文件權限
                        for root, dirs, files in os.walk(db_path):
                            os.chmod(root, 0o755)
                            for dir in dirs:
                                os.chmod(os.path.join(root, dir), 0o755)
                            for file in files:
                                os.chmod(os.path.join(root, file), 0o644)
                    
                    shutil.rmtree(db_path)
                    logger.info(f"成功刪除資料庫目錄: {db_path}")
                except Exception as e:
                    logger.error(f"刪除資料庫目錄 {db_path} 時出錯: {e}", exc_info=True)
                    self._update_status("error", 0, f"刪除資料庫目錄失敗: {e}")
                    return None
            
            # 確保資料庫目錄存在
            try:
                os.makedirs(db_path, exist_ok=True)
                # 設置適當的權限
                os.chmod(db_path, 0o755)
                logger.info(f"已確保資料庫目錄存在/已創建: {db_path}")
            except OSError as e:
                logger.error(f"創建資料庫目錄 {db_path} 失敗: {e}", exc_info=True)
                self._update_status("error", 0, f"創建資料庫目錄失敗: {e}")
                return None
            
            # 2. 準備文本分割器
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                add_start_index=True,
            )
            
            # 3. 檢查並準備 PDF 資料夾
            if not os.path.exists(pdf_folder):
                logger.warning(f"PDF資料夾不存在: {pdf_folder}")
                try:
                    os.makedirs(pdf_folder, exist_ok=True)
                    logger.info(f"已成功創建PDF資料夾: {pdf_folder}")
                except OSError as e:
                    logger.error(f"創建PDF資料夾 {pdf_folder} 失敗: {e}", exc_info=True)
                    self._update_status("error", 5, f"創建PDF資料夾失敗: {e}")
                    return None
            
            # 4. 獲取所有 PDF 檔案
            files_to_process = self._get_all_pdf_files(pdf_folder)
            
            if not files_to_process:
                logger.warning(f"在 '{pdf_folder}' 中沒有找到任何 PDF 檔案")
                # 創建空的向量資料庫
                try:
                    # 確保目錄權限
                    if not os.path.exists(db_path):
                        os.makedirs(db_path, mode=0o755)
                    else:
                        os.chmod(db_path, 0o755)
                    
                    embedding_function = HuggingFaceEmbeddings(model_name=emb_model)
                    vector_db = Chroma(
                        embedding_function=embedding_function,
                        persist_directory=db_path
                    )
                    vector_db.persist()
                    
                    # 設置創建後的權限
                    if os.path.exists(db_path):
                        for root, dirs, files in os.walk(db_path):
                            os.chmod(root, 0o755)
                            for file in files:
                                os.chmod(os.path.join(root, file), 0o644)
                    
                    logger.info(f"已在 '{db_path}' 創建了一個空的向量資料庫")
                    self._update_status("completed", 100, "創建空的向量資料庫完成")
                    self.db = vector_db  # 設置實例變數
                    self.embedding_function = embedding_function  # 設置實例變數
                    return vector_db
                except Exception as e:
                    logger.error(f"創建嵌入模型或空的向量資料庫時出錯: {e}", exc_info=True)
                    self._update_status("error", 5, f"創建空資料庫失敗: {e}")
                    return None
            
            logger.info(f"準備處理 {len(files_to_process)} 個PDF檔案: {files_to_process}")
            self._rebuild_status["total_files"] = len(files_to_process)
            
            # 5. 載入並分割所有 PDF 文件
            all_document_chunks = []
            for i, pdf_file_name in enumerate(files_to_process, 1):
                progress = 10 + int(70 * i / len(files_to_process))
                self._update_status("processing", progress, f"處理 PDF ({i}/{len(files_to_process)}): {pdf_file_name}")
                
                file_path = os.path.join(pdf_folder, pdf_file_name)
                try:
                    logger.info(f"正在處理檔案 ({i}/{len(files_to_process)}): {pdf_file_name}")
                    loader = PyPDFLoader(file_path, extract_images=False)
                    pages = loader.load()
                    chunked_pages = text_splitter.split_documents(pages)
                    
                    # 為每個文本塊添加詳細的元數據
                    for chunk in chunked_pages:
                        chunk.metadata["source_pdf"] = pdf_file_name
                        chunk.metadata["source_file_name"] = pdf_file_name
                        chunk.metadata["chunk_index"] = chunked_pages.index(chunk)
                        chunk.metadata["total_chunks_in_file"] = len(chunked_pages)
                    
                    all_document_chunks.extend(chunked_pages)
                    logger.debug(f"檔案 '{pdf_file_name}' 被成功處理並分割成 {len(chunked_pages)} 個文本塊")
                    self._rebuild_status["processed_files"] = i
                except Exception as e:
                    logger.error(f"處理PDF檔案 '{file_path}' 時發生錯誤: {e}", exc_info=True)
            
            # 6. 如果所有 PDF 都處理失敗
            if not all_document_chunks:
                logger.warning("所有PDF檔案均處理失敗或沒有可提取的內容")
                try:
                    # 確保目錄權限
                    if not os.path.exists(db_path):
                        os.makedirs(db_path, mode=0o755)
                    else:
                        os.chmod(db_path, 0o755)
                    
                    embedding_function = HuggingFaceEmbeddings(model_name=emb_model)
                    vector_db = Chroma(embedding_function=embedding_function, persist_directory=db_path)
                    vector_db.persist()
                    
                    # 設置創建後的權限
                    if os.path.exists(db_path):
                        for root, dirs, files in os.walk(db_path):
                            os.chmod(root, 0o755)
                            for file in files:
                                os.chmod(os.path.join(root, file), 0o644)
                    
                    logger.info(f"已在 '{db_path}' 創建空的向量資料庫 (因無成功處理的文本塊)")
                    self._update_status("completed", 100, "創建空資料庫完成 (無成功處理的文本)")
                    self.db = vector_db  # 設置實例變數
                    self.embedding_function = embedding_function  # 設置實例變數
                    return vector_db
                except Exception as e:
                    logger.error(f"創建嵌入模型或空向量資料庫時出錯: {e}", exc_info=True)
                    self._update_status("error", 80, f"創建空資料庫失敗: {e}")
                    return None
            
            logger.info(f"成功從所有PDF檔案中處理了 {len(all_document_chunks)} 個文本塊")
            
            # 7. 創建嵌入模型
            try:
                self._update_status("processing", 85, f"創建嵌入模型: {emb_model}")
                logger.info(f"正在使用嵌入模型 '{emb_model}' 創建文本嵌入...")
                embedding_function = HuggingFaceEmbeddings(model_name=emb_model)
            except Exception as e:
                logger.error(f"創建嵌入模型 '{emb_model}' 失敗: {e}", exc_info=True)
                self._update_status("error", 85, f"創建嵌入模型失敗: {e}")
                return None
            
            # 8. 建立向量資料庫
            try:
                self._update_status("processing", 90, "創建向量資料庫")
                logger.info(f"正在創建向量資料庫並保存到 '{db_path}'...")
                
                # 再次確保目錄權限
                if not os.path.exists(db_path):
                    os.makedirs(db_path, mode=0o755)
                else:
                    os.chmod(db_path, 0o755)
                
                vector_db = Chroma.from_documents(
                    documents=all_document_chunks,
                    embedding=embedding_function,
                    persist_directory=db_path
                )
                
                # 設置創建後的權限
                if os.path.exists(db_path):
                    for root, dirs, files in os.walk(db_path):
                        os.chmod(root, 0o755)
                        for file in files:
                            os.chmod(os.path.join(root, file), 0o644)
                
                vector_db.persist()
                self.db = vector_db
                self.embedding_function = embedding_function
                
                logger.info("向量資料庫處理完成!")
                self._update_status("completed", 100, "向量資料庫重建完成")
                return vector_db
            except Exception as e:
                logger.error(f"創建向量資料庫失敗: {e}", exc_info=True)
                self._update_status("error", 90, f"創建向量資料庫失敗: {e}")
                return None
                
        finally:
            self._rebuilding = False
    
    def connect_db(self) -> Optional[Chroma]:
        """連接到現有的向量資料庫"""
        with self._lock:
            if self.db is not None:
                return self.db
            
            if not os.path.exists(self.db_path) or not os.listdir(self.db_path):
                logger.warning(f"向量資料庫路徑 '{self.db_path}' 不存在或為空")
                return None
            
            try:
                logger.info(f"嘗試連接到向量資料庫: {self.db_path}")
                
                if self.embedding_function is None:
                    self.embedding_function = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
                
                self.db = Chroma(
                    embedding_function=self.embedding_function,
                    persist_directory=self.db_path
                )
                
                # 驗證資料庫
                try:
                    collection = self.db._collection
                    count = collection.count() if collection else 0
                    logger.info(f"成功連接到向量資料庫: {self.db_path} (記錄數: {count})")
                except Exception as e:
                    logger.warning(f"獲取資料庫記錄數時出錯: {e}")
                
                return self.db
            except Exception as e:
                logger.error(f"連接資料庫失敗: {e}", exc_info=True)
                self.db = None
                return None
    
    def search(self, query: str, k: Optional[int] = None, source_files: Optional[List[str]] = None) -> List[Document]:
        """搜索向量資料庫
        
        Args:
            query: 查詢字串
            k: 返回結果數量，None 表示使用預設值
            source_files: 指定要搜索的源文件列表，None 表示搜索所有文件
            
        Returns:
            搜索結果列表
        """
        if k is None:
            k = self.default_top_k
            
        if self.db is None:
            self.connect_db()
        
        if self.db is None:
            logger.warning("資料庫未連接，無法執行搜索")
            return []
        
        try:
            if source_files is None:
                # 搜索所有文件
                results = self.db.similarity_search(query, k=k)
                logger.debug(f"搜索查詢 '{query}' 返回 {len(results)} 個結果（所有文件）")
            else:
                # 搜索指定文件
                where_clause = {"source_file_name": {"$in": source_files}}
                results = self.db.similarity_search(query, k=k, filter=where_clause)
                logger.debug(f"搜索查詢 '{query}' 返回 {len(results)} 個結果（指定文件: {source_files}）")
            
            return results
        except Exception as e:
            logger.error(f"執行搜索時發生錯誤: {e}", exc_info=True)
            return []
    
    def get_available_source_files(self) -> List[str]:
        """獲取資料庫中所有可用的源文件列表"""
        if self.db is None:
            self.connect_db()
        
        if self.db is None:
            logger.warning("資料庫未連接，無法獲取源文件列表")
            return []
        
        try:
            # 獲取所有文檔的元數據
            results = self.db._collection.get(include=['metadatas'])
            if not results or not results.get('metadatas'):
                return []
            
            source_files = set()
            for metadata in results['metadatas']:
                if metadata and 'source_file_name' in metadata:
                    source_files.add(metadata['source_file_name'])
            
            source_files_list = sorted(list(source_files))
            logger.debug(f"找到 {len(source_files_list)} 個源文件")
            return source_files_list
        except Exception as e:
            logger.error(f"獲取源文件列表時發生錯誤: {e}", exc_info=True)
            return []
    
    def get_status(self) -> dict:
        """獲取資料庫狀態和重建進度"""
        with self._lock:
            db_exists = os.path.exists(self.db_path) and bool(os.listdir(self.db_path))
            db_connected = self.db is not None
            
            # 獲取 PDF 檔案資訊
            pdf_files = []
            pdf_count = 0
            
            if os.path.exists(self.pdf_folder):
                try:
                    pdf_files = [f for f in os.listdir(self.pdf_folder) if f.lower().endswith('.pdf')]
                    pdf_count = len(pdf_files)
                except Exception as e:
                    logger.error(f"讀取 PDF 資料夾時出錯: {e}")
            
            # 獲取資料庫記錄數
            record_count = 0
            if db_connected and self.db:
                try:
                    record_count = self.db._collection.count() if hasattr(self.db, '_collection') and self.db._collection else 0
                except Exception as e:
                    logger.error(f"獲取資料庫記錄數時出錯: {e}")
            
            # 獲取資料庫中的源文件列表
            db_source_files = []
            if db_connected:
                db_source_files = self.get_available_source_files()
            
            return {
                "db_path": self.db_path,
                "db_exists": db_exists,
                "db_connected": db_connected,
                "pdf_folder": self.pdf_folder,
                "pdf_count": pdf_count,
                "pdf_files": pdf_files,
                "db_source_files": db_source_files,
                "db_source_files_count": len(db_source_files),
                "embedding_model": self.embedding_model_name,
                "record_count": record_count,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "default_top_k": self.default_top_k,
                "rebuilding": self._rebuilding,
                "rebuild_status": self._rebuild_status.copy()
            }
    
    def get_rebuild_status(self) -> dict:
        """獲取重建狀態"""
        with self._lock:
            return self._rebuild_status.copy()
    
    def _get_all_pdf_files(self, pdf_folder: str) -> List[str]:
        """獲取資料夾中所有的 PDF 檔案"""
        try:
            all_folder_contents = os.listdir(pdf_folder)
            pdf_files = [f for f in all_folder_contents if f.lower().endswith(".pdf")]
            logger.info(f"在 '{pdf_folder}' 中找到 {len(pdf_files)} 個PDF檔案")
            return pdf_files
        except OSError as e:
            logger.error(f"讀取PDF資料夾 '{pdf_folder}' 內容失敗: {e}", exc_info=True)
            return []
    
    def _update_status(self, status: str, progress: int, message: str):
        """更新重建狀態"""
        with self._lock:
            self._rebuild_status.update({
                "status": status,
                "progress": progress,
                "message": message,
                "update_time": datetime.now().isoformat()
            })
            
            if status == "started":
                self._rebuild_status["start_time"] = datetime.now().isoformat()
                self._rebuild_status["processed_files"] = 0
            
            logger.info(f"重建狀態更新: {status}, {progress}%, {message}")