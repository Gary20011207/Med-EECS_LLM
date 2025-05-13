# core/db_manager.py 
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
import shutil
import logging
from typing import List, Optional
import threading
from datetime import datetime

# 導入配置
try:
    from config import (
        EMBEDDINGS_MODEL_NAME,
        PDF_FOLDER,
        DB_PATH,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        RAG_TOP_K,
        config_manager  # 新增：用於監聽配置變更
    )
except ImportError:
    # 如果無法導入配置，使用預設值
    EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    PDF_FOLDER = "./PDFS"
    DB_PATH = "./VectorDB"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RAG_TOP_K = 5
    config_manager = None

logger = logging.getLogger(__name__)

class VectorDBManager:
    def __init__(self):
        self._update_config_values()
        self.db = None
        self.embedding_function = None
        self._rebuilding = False
        self._rebuild_status = {
            "status": "idle",
            "progress": 0,
            "message": "",
            "start_time": None,
            "total_files": 0,
            "processed_files": 0
        }
        # 初始化 embedding function
        self._init_embedding_function()

    def _update_config_values(self):
        self.pdf_folder = PDF_FOLDER
        self.db_path = DB_PATH
        self.embedding_model_name = EMBEDDINGS_MODEL_NAME
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
        self.default_top_k = RAG_TOP_K

    def _init_embedding_function(self):
        """初始化 embedding function"""
        if self.embedding_model_name:
            self.embedding_function = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        else:
            logger.warning("未指定 embedding 模型，將無法進行向量化")
            self.embedding_function = None

    def reload_config(self):
        """重新載入配置"""
        logger.info("VectorDBManager: 重新載入配置")
        old_embedding_model = self.embedding_model_name
        self._update_config_values()
        
        # 如果 embedding 模型改變，需要重新初始化
        if old_embedding_model != self.embedding_model_name:
            logger.info(f"Embedding 模型已改變: {old_embedding_model} -> {self.embedding_model_name}")
            self._init_embedding_function()
            
            # 如果資料庫已連接且模型改變，需要重建資料庫
            if self.db is not None:
                logger.warning("Embedding 模型改變，建議重建向量資料庫")

    def connect_db(self) -> Optional[Chroma]:
            if self.db is not None:
                return self.db

            if not os.path.exists(self.db_path) or not os.listdir(self.db_path):
                logger.warning("資料庫不存在或為空，開始重建")
                return self.reset_and_rebuild_db()

            try:
                self.db = Chroma(embedding_function=self.embedding_function, persist_directory=self.db_path)

                existing_sources = self.get_available_source_files()
                current_pdfs = self._get_all_pdf_files(self.pdf_folder)

                if set(current_pdfs) != set(existing_sources):
                    logger.info("PDF 清單異動，開始重建資料庫")
                    return self.reset_and_rebuild_db()

                logger.info("成功連接到資料庫")
                return self.db

            except Exception as e:
                logger.error(f"連接資料庫失敗: {e}", exc_info=True)
                return None

    def reset_and_rebuild_db(self) -> Optional[Chroma]:
            if self._rebuilding:
                logger.warning("資料庫重建中，請稍後")
                return None
            self._rebuilding = True
            self.db = None

            try:
                logger.info("開始重建向量資料庫")
                self._update_status("started", 0, "初始化中")

                if os.path.exists(self.db_path):
                    shutil.rmtree(self.db_path)
                os.makedirs(self.db_path, exist_ok=False)

                pdf_files = self._get_all_pdf_files(self.pdf_folder)
                if not pdf_files:
                    return self._create_empty_db()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    length_function=len,
                    add_start_index=True
                )

                all_chunks = []
                for i, pdf_file in enumerate(pdf_files):
                    path = os.path.join(self.pdf_folder, pdf_file)
                    try:
                        pages = PyPDFLoader(path).load()
                        chunks = text_splitter.split_documents(pages)
                        for idx, chunk in enumerate(chunks):
                            chunk.metadata.update({
                                "source_file_name": pdf_file,
                                "chunk_index": idx,
                                "total_chunks_in_file": len(chunks)
                            })
                        all_chunks.extend(chunks)
                    except Exception as e:
                        logger.warning(f"無法處理 {pdf_file}: {e}")

                if not all_chunks:
                    return self._create_empty_db()

                self.db = Chroma.from_documents(all_chunks, self.embedding_function, persist_directory=self.db_path)

                self._update_status("completed", 100, "重建完成")
                logger.info("向量資料庫重建完成")
                return self.db

            except Exception as e:
                logger.error(f"重建資料庫失敗: {e}", exc_info=True)
                self._update_status("error", 100, f"重建失敗: {e}")
                return None

            finally:
                self._rebuilding = False

    def _create_empty_db(self) -> Optional[Chroma]:
        try:
            self.db = Chroma(embedding_function=self.embedding_function, persist_directory=self.db_path)
            self._update_status("completed", 100, "已建立空資料庫")
            return self.db
        except Exception as e:
            logger.error(f"建立空資料庫失敗: {e}", exc_info=True)
            return None

    def get_available_source_files(self) -> List[str]:
        if self.db is None:
            return []
        try:
            results = self.db._collection.get(include=['metadatas'])
            return sorted({m['source_file_name'] for m in results['metadatas'] if 'source_file_name' in m})
        except Exception as e:
            logger.warning(f"讀取資料庫元數據失敗: {e}")
            return []

    def _get_all_pdf_files(self, folder: str) -> List[str]:
        try:
            return sorted(f for f in os.listdir(folder) if f.lower().endswith(".pdf"))
        except Exception as e:
            logger.warning(f"讀取 PDF 目錄失敗: {e}")
            return []

    def _update_status(self, status: str, progress: int, message: str):
            self._rebuild_status.update({
                "status": status,
                "progress": progress,
                "message": message,
                "update_time": datetime.now().isoformat(),
                "start_time": self._rebuild_status.get("start_time") or datetime.now().isoformat()
            })
   
    def search(self, query: str, k: Optional[int] = None, source_files: Optional[List[str]] = None) -> List[Document]:
        if k is None:
            k = self.default_top_k

        if self.db is None:
            self.connect_db()

        if self.db is None:
            logger.warning("資料庫未連接，無法執行搜索")
            return []

        try:
            if source_files is None:
                results = self.db.similarity_search(query, k=k)
                logger.info(f"查詢 '{query}' 返回 {len(results)} 筆結果（所有文件）")
            else:
                filter_clause = {"source_file_name": {"$in": source_files}}
                results = self.db.similarity_search(query, k=k, filter=filter_clause)
                logger.info(f"查詢 '{query}' 返回 {len(results)} 筆結果（指定文件）")
            return results
        except Exception as e:
            logger.error(f"搜尋發生錯誤: {e}", exc_info=True)
            return []
