import os
import shutil
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 設定 logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reset_and_rebuild_vectordb(
    pdf_folder: str = "./PDFS",
    db_path: str = "./VectorDB",
    emb_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    force_reset: bool = True
) -> Chroma | None:
    """
    重置並重建向量資料庫。

    參數:
        pdf_folder (str): PDF文件所在的資料夾路徑
        db_path (str): 向量資料庫儲存路徑
        emb_model (str): 嵌入模型名稱
        chunk_size (int): 分割文本的塊大小
        chunk_overlap (int): 分割文本的重疊大小
        force_reset (bool): 是否強制重置，即使資料庫已存在

    返回:
        Chroma: 重建後的向量資料庫實例，如果發生嚴重錯誤導致無法建立則返回 None 或引發異常。
    
    可能引發的錯誤:
        RuntimeError: 如果 force_reset 為 True 但刪除現有資料庫失敗。
        IOError: 如果 PDF 資料夾無法創建或訪問 (雖然已嘗試創建)。
    """
    if force_reset and os.path.exists(db_path):
        logger.info(f"正在刪除現有向量資料庫: {db_path}")
        try:
            shutil.rmtree(db_path)
            logger.info(f"成功刪除資料庫目錄: {db_path}")
        except Exception as e:
            logger.error(f"刪除資料庫目錄 {db_path} 時出錯: {e}")
            raise RuntimeError(f"無法刪除資料庫目錄 {db_path}，重建終止。") from e
            
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    if not os.path.exists(pdf_folder):
        logger.warning(f"PDF資料夾不存在: {pdf_folder}")
        try:
            os.makedirs(pdf_folder, exist_ok=True)
            logger.info(f"已創建PDF資料夾: {pdf_folder}")
        except OSError as e:
            logger.error(f"創建PDF資料夾 {pdf_folder} 失敗: {e}")
            return None
            
    try:
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    except OSError as e:
        logger.error(f"讀取PDF資料夾 {pdf_folder} 內容失敗: {e}")
        return None

    if not pdf_files:
        logger.warning(f"未在 {pdf_folder} 中找到任何PDF文件。將創建/載入一個空的向量資料庫。")
        try:
            embed = HuggingFaceEmbeddings(model_name=emb_model)
            vector_db = Chroma(embedding_function=embed, persist_directory=db_path)
            logger.info(f"已在 {db_path} 創建/載入空的向量資料庫。")
            return vector_db
        except Exception as e:
            logger.error(f"創建嵌入模型或空向量資料庫時出錯: {e}")
            return None
    
    logger.info(f"找到 {len(pdf_files)} 個PDF文件，開始處理...")
    
    docs = []
    for i, f in enumerate(pdf_files, 1):
        try:
            logger.info(f"處理 ({i}/{len(pdf_files)}): {f}")
            file_path = os.path.join(pdf_folder, f)
            loader = PyPDFLoader(file_path)
            loaded_docs_pages = loader.load()
            chunked_docs = splitter.split_documents(loaded_docs_pages)
            for chunk in chunked_docs:
                chunk.metadata["source_pdf"] = f
                docs.append(chunk)
            logger.debug(f"文件 {f} 被分割成 {len(chunked_docs)} 個文本塊。")
        except Exception as e:
            logger.error(f"處理PDF文件 {f} 時出錯: {e}")
    
    if not docs:
        logger.warning("所有PDF均處理失敗或沒有可處理的內容，將創建/載入一個空的向量資料庫。")
        try:
            embed = HuggingFaceEmbeddings(model_name=emb_model)
            vector_db = Chroma(embedding_function=embed, persist_directory=db_path)
            logger.info(f"已在 {db_path} 創建/載入空的向量資料庫 (因無成功處理的文本塊)。")
            return vector_db
        except Exception as e:
            logger.error(f"創建嵌入模型或空向量資料庫時出錯 (因無成功處理的文本塊): {e}")
            return None

    logger.info(f"成功處理了 {len(docs)} 個文本塊。")
    
    try:
        logger.info(f"使用模型 {emb_model} 創建嵌入...")
        embed = HuggingFaceEmbeddings(model_name=emb_model)
    except Exception as e:
        logger.error(f"創建嵌入模型 {emb_model} 失敗: {e}")
        return None

    try:
        logger.info(f"創建向量資料庫並保存到 {db_path}...")
        vectordb = Chroma.from_documents(
            documents=docs, 
            embedding=embed, 
            persist_directory=db_path
        )
        logger.info("向量資料庫重置與重建完成!")
        return vectordb
    except Exception as e:
        logger.error(f"從文檔創建向量資料庫並保存到 {db_path} 時失敗: {e}")
        return None


if __name__ == "__main__":
    # 基礎日誌配置，如果此腳本獨立運行
    # 在實際應用中，通常在主程序 app.py 中統一配置日誌
    # 檢查是否已經有 handlers，避免重複添加 (例如被 app.py 導入時)
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, # INFO 級別適用於一般情況
                            # DEBUG 級別可以看到 langchain 等庫更詳細的日誌
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("="*50)
    logger.info("開始執行 RAG_build.py 腳本 (使用預設資料夾 ./PDFS 和 ./VectorDB)")
    logger.info("="*50)
    
    # 確保 ./PDFS 資料夾存在，如果不存在則提示用戶
    default_pdf_folder = "./PDFS"
    if not os.path.exists(default_pdf_folder):
        logger.warning(f"預設PDF資料夾 {default_pdf_folder} 不存在。")
        logger.warning(f"請先創建 {default_pdf_folder} 並放入PDF文件，或者腳本會創建一個空的資料夾。")
        # 函數內部已有創建資料夾的邏輯，此處僅為提示
    elif not any(f.endswith(".pdf") for f in os.listdir(default_pdf_folder)):
        logger.warning(f"預設PDF資料夾 {default_pdf_folder} 中沒有找到任何 PDF 文件。")
        logger.warning("腳本將會嘗試創建一個空的向量資料庫。")


    try:
        # 直接調用函數，使用其預設的 pdf_folder 和 db_path
        # 預設 force_reset=True，將會清空並重建 ./VectorDB
        logger.info(f"將使用預設參數重建資料庫: pdf_folder='./PDFS', db_path='./VectorDB', force_reset=True")
        logger.info("注意：這將會修改 ./VectorDB 中的內容！")
        
        vectordb_instance = reset_and_rebuild_vectordb(force_reset=True) #明確指定 force_reset
        
        if vectordb_instance:
            count = vectordb_instance._collection.count()
            logger.info(f"向量資料庫實例操作成功。")
            logger.info(f"向量資料庫 (./VectorDB) 包含 {count} 條記錄。")
            
            if count > 0:
                try:
                    # 執行一個簡單的查詢來驗證
                    logger.info("執行簡單相似性搜索測試...")
                    results = vectordb_instance.similarity_search("ERAS", k=1) # 假設您的PDF內容與ERAS相關
                    if results:
                        logger.info(f"相似性搜索測試結果 (取1個，內容片段): {results[0].page_content[:200]}...")
                        logger.info(f"來源PDF: {results[0].metadata.get('source_pdf', '未知')}")
                    else:
                        logger.info("相似性搜索未返回結果，但資料庫非空。")
                except Exception as e:
                    logger.error(f"相似性搜索測試失敗: {e}")
            else:
                logger.info("向量資料庫為空，跳過相似性搜索測試。")
        else:
            logger.error("向量資料庫實例操作失敗 (函數返回 None 或引發了未捕獲的異常)。")

    except RuntimeError as e:
        logger.error(f"腳本執行過程中發生運行時錯誤: {e}")
    except Exception as e:
        logger.error(f"腳本執行過程中發生未預期錯誤: {e}", exc_info=True) # exc_info=True 會記錄堆疊追蹤
    finally:
        logger.info("="*50)
        logger.info("RAG_build.py 腳本執行完畢。")
        logger.info("="*50)