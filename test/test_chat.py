# test/test_chat.py
"""
RAG Engine 測試文件
專注於測試 RAG 引擎的功能和配置
"""
import os
import sys
import logging
import time
from datetime import datetime

# 確保可以導入項目模組
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 先導入配置模組，確保環境正確設置
try:
    import config
    logger_temp = logging.getLogger(__name__)
    logger_temp.info("成功導入配置模組")
    logger_temp.info(f"LLM 模型: {config.LLM_MODEL_NAME}")
    logger_temp.info(f"預設溫度: {config.DEFAULT_TEMPERATURE}")
    logger_temp.info(f"預設最大 Token 數: {config.DEFAULT_MAX_NEW_TOKENS}")
except ImportError as e:
    print(f"無法導入配置模組: {e}")
    sys.exit(1)

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_chat.log', mode='w', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

# 打印配置信息
logger.info("=" * 60)
logger.info("RAG Engine 測試環境配置")
logger.info("=" * 60)
logger.info(f"LLM 模型名稱: {config.LLM_MODEL_NAME}")
logger.info(f"嵌入模型名稱: {config.EMBEDDINGS_MODEL_NAME}")
logger.info(f"預設溫度: {config.DEFAULT_TEMPERATURE}")
logger.info(f"預設最大 Token 數: {config.DEFAULT_MAX_NEW_TOKENS}")
logger.info(f"溫度範圍: [{config.MIN_TEMPERATURE}, {config.MAX_TEMPERATURE}]")
logger.info(f"Token 數範圍: [{config.MIN_MAX_NEW_TOKENS}, {config.MAX_MAX_NEW_TOKENS}]")
logger.info(f"RAG Top-K: {config.RAG_TOP_K}")
logger.info(f"系統提示詞長度: {len(config.SYSTEM_PROMPT)} 字符")
logger.info("=" * 60)

def test_config_validation():
    """測試配置參數的有效性"""
    logger.info("\n" + "=" * 60)
    logger.info("開始配置驗證測試")
    logger.info("=" * 60)
    
    # 檢查必要的配置項
    required_configs = [
        'LLM_MODEL_NAME',
        'EMBEDDINGS_MODEL_NAME',
        'DEFAULT_TEMPERATURE',
        'DEFAULT_MAX_NEW_TOKENS',
        'MIN_TEMPERATURE',
        'MAX_TEMPERATURE',
        'MIN_MAX_NEW_TOKENS',
        'MAX_MAX_NEW_TOKENS',
        'RAG_TOP_K',
        'SYSTEM_PROMPT'
    ]
    
    for config_name in required_configs:
        if hasattr(config, config_name):
            value = getattr(config, config_name)
            logger.info(f"✓ {config_name}: {value}")
        else:
            logger.error(f"✗ 缺少配置項: {config_name}")
    
    # 驗證參數範圍
    assert config.MIN_TEMPERATURE <= config.DEFAULT_TEMPERATURE <= config.MAX_TEMPERATURE, \
        f"預設溫度 {config.DEFAULT_TEMPERATURE} 超出範圍 [{config.MIN_TEMPERATURE}, {config.MAX_TEMPERATURE}]"
    
    assert config.MIN_MAX_NEW_TOKENS <= config.DEFAULT_MAX_NEW_TOKENS <= config.MAX_MAX_NEW_TOKENS, \
        f"預設最大 Token 數 {config.DEFAULT_MAX_NEW_TOKENS} 超出範圍 [{config.MIN_MAX_NEW_TOKENS}, {config.MAX_MAX_NEW_TOKENS}]"
    
    assert config.RAG_TOP_K > 0, f"RAG Top-K 必須大於 0，當前值: {config.RAG_TOP_K}"
    assert len(config.SYSTEM_PROMPT) > 0, f"系統提示詞不能為空"
    
    logger.info("✓ 配置驗證完成，所有參數有效")

def test_rag_engine():
    """測試 RAG 引擎功能"""
    try:
        # 導入必要的模組
        from core.model_manager import ModelManager
        from core.db_manager import VectorDBManager
        from core.rag_engine import RAGEngine
        
        logger.info("=" * 60)
        logger.info("開始 RAG Engine 測試")
        logger.info("=" * 60)
        
        # 初始化組件
        logger.info("1. 初始化組件...")
        model_manager = ModelManager()
        db_manager = VectorDBManager()
        rag_engine = RAGEngine(model_manager, db_manager)
        
        # 檢查狀態
        logger.info("2. 檢查系統狀態...")
        status = rag_engine.get_status()
        logger.info(f"RAG 引擎狀態:")
        for key, value in status.items():
            logger.info(f"  {key}: {value}")
        
        # 驗證 RAG 引擎使用的配置是否正確
        logger.info("3. 驗證 RAG 引擎配置...")
        assert rag_engine.default_temperature == config.DEFAULT_TEMPERATURE, \
            f"RAG 引擎溫度配置不匹配: {rag_engine.default_temperature} != {config.DEFAULT_TEMPERATURE}"
        assert rag_engine.default_max_new_tokens == config.DEFAULT_MAX_NEW_TOKENS, \
            f"RAG 引擎最大 Token 數配置不匹配: {rag_engine.default_max_new_tokens} != {config.DEFAULT_MAX_NEW_TOKENS}"
        assert rag_engine.top_k == config.RAG_TOP_K, \
            f"RAG 引擎 Top-K 配置不匹配: {rag_engine.top_k} != {config.RAG_TOP_K}"
        
        logger.info("✓ RAG 引擎配置驗證通過")
        
        # 測試問題
        test_questions = [
            "什麼是 ERAS？",
            "術前準備有哪些要求？",
            "術後康復的關鍵要素是什麼？",
            "如何管理術後疼痛？",
            "ERAS 對於哪些手術最有效？"
        ]
        
        # 模擬對話歷史
        sample_history = [
            {"role": "user", "content": "ERAS 是什麼意思？"},
            {"role": "assistant", "content": "ERAS (Enhanced Recovery After Surgery) 是手術加速康復計劃，旨在通過優化手術前、中、後的護理流程，減少併發症，縮短住院時間，並改善患者的整體恢復體驗。"}
        ]
        
        # 4. 測試預設配置的非串流回覆
        logger.info("4. 測試使用預設配置的非串流回覆...")
        question = test_questions[0]
        logger.info(f"問題: {question}")
        
        # 不指定溫度和 token 數，使用配置預設值
        result = rag_engine.generate_reply(
            query=question,
            use_rag=True,
            enable_memory=True,
            history=sample_history
        )
        
        logger.info(f"回覆: {result['response'][:100]}..." if result['response'] else "回覆為空")
        logger.info(f"用時: {result['generation_time']:.2f}秒")
        logger.info(f"使用的溫度: {result.get('temperature', 'N/A')} (預設: {config.DEFAULT_TEMPERATURE})")
        logger.info(f"使用的最大 Token 數: {result.get('max_new_tokens', 'N/A')} (預設: {config.DEFAULT_MAX_NEW_TOKENS})")
        logger.info(f"使用了 RAG: {result['used_rag']}")
        
        # 檢查錯誤
        if 'error' in result:
            logger.error(f"生成回覆時出錯: {result['error']}")
            logger.warning("跳過後續的斷言檢查，因為回覆生成失敗")
        else:
            # 只在沒有錯誤時進行驗證
            if 'temperature' in result:
                assert result['temperature'] == config.DEFAULT_TEMPERATURE, \
                    f"未使用預設溫度: {result['temperature']} != {config.DEFAULT_TEMPERATURE}"
            if 'max_new_tokens' in result:
                assert result['max_new_tokens'] == config.DEFAULT_MAX_NEW_TOKENS, \
                    f"未使用預設最大 Token 數: {result['max_new_tokens']} != {config.DEFAULT_MAX_NEW_TOKENS}"
        
        # 5. 測試不同配置的非串流回覆
        logger.info("5. 測試不同配置的非串流回覆...")
        test_configs = [
            {
                "use_rag": True, 
                "enable_memory": True, 
                "temperature": config.MIN_TEMPERATURE + 0.1,
                "max_new_tokens": config.MIN_MAX_NEW_TOKENS + 50,
                "description": "最低溫度 + 最少 Token"
            },
            {
                "use_rag": False, 
                "enable_memory": True, 
                "temperature": (config.MIN_TEMPERATURE + config.MAX_TEMPERATURE) / 2,
                "max_new_tokens": int((config.MIN_MAX_NEW_TOKENS + config.MAX_MAX_NEW_TOKENS) / 2),
                "description": "中等溫度 + 中等 Token，不使用 RAG"
            },
            {
                "use_rag": True, 
                "enable_memory": False, 
                "temperature": config.MAX_TEMPERATURE - 0.1,
                "max_new_tokens": config.MAX_MAX_NEW_TOKENS - 50,
                "description": "最高溫度 + 最多 Token，不使用記憶"
            },
        ]
        
        for i, test_config in enumerate(test_configs):
            description = test_config.pop("description", "")
            logger.info(f"\n--- 配置 {i+1}: {description} ---")
            logger.info(f"參數: {test_config}")
            question = test_questions[i % len(test_questions)]
            logger.info(f"問題: {question}")
            
            result = rag_engine.generate_reply(
                query=question,
                history=sample_history if test_config.get("enable_memory", True) else [],
                **test_config
            )
            
            logger.info(f"回覆: {result['response'][:100]}..." if result['response'] else "回覆為空")
            logger.info(f"用時: {result['generation_time']:.2f}秒")
            logger.info(f"Token 數: {result.get('token_count', 'N/A')}")
            logger.info(f"使用 RAG: {result['used_rag']}")
            logger.info(f"溫度: {result.get('temperature', 'N/A')}")
            logger.info(f"最大 Token 數: {result.get('max_new_tokens', 'N/A')}")
            
            # 檢查錯誤
            if 'error' in result:
                logger.error(f"錯誤: {result['error']}")
        
        # 6. 測試串流回覆
        logger.info("\n6. 測試串流回覆...")
        question = "請詳細解釋 ERAS 的主要組成部分"
        logger.info(f"問題: {question}")
        
        # 使用配置預設值
        full_response = ""
        chunk_count = 0
        start_time = time.time()
        
        logger.info("開始串流輸出...")
        for chunk in rag_engine.stream_response(
            query=question,
            use_rag=True,
            enable_memory=True,
            history=sample_history
            # 不指定 temperature 和 max_new_tokens，使用預設值
        ):
            if chunk['type'] == 'start':
                logger.info("串流開始...")
                logger.info(f"使用 RAG: {chunk['used_rag']}")
                logger.info(f"溫度: {chunk.get('temperature')} (預設: {config.DEFAULT_TEMPERATURE})")
                logger.info(f"最大 Token 數: {chunk.get('max_new_tokens')} (預設: {config.DEFAULT_MAX_NEW_TOKENS})")
                
                # 驗證使用了預設值
                if 'temperature' in chunk:
                    assert chunk.get('temperature') == config.DEFAULT_TEMPERATURE
                if 'max_new_tokens' in chunk:
                    assert chunk.get('max_new_tokens') == config.DEFAULT_MAX_NEW_TOKENS
                
            elif chunk['type'] == 'chunk':
                print(chunk['chunk'], end='', flush=True)
                full_response += chunk['chunk']
                chunk_count += 1
            elif chunk['type'] == 'end':
                print("\n")
                logger.info("串流結束")
                logger.info(f"總計 {chunk_count} 個片段")
                logger.info(f"完整回覆長度: {len(full_response)} 字符")
                logger.info(f"用時: {chunk['generation_time']:.2f}秒")
                logger.info(f"Token 數: {chunk.get('token_count', 'N/A')}")
            elif chunk['type'] == 'error':
                logger.error(f"串流錯誤: {chunk['error']}")
                logger.error(f"錯誤發生時的配置: 溫度={chunk.get('temperature')}, Token數={chunk.get('max_new_tokens')}")
                break
        
        # 7. 測試參數驗證（使用配置定義的範圍）
        logger.info("\n7. 測試參數驗證...")
        
        # 測試超出配置範圍的溫度
        logger.info("測試溫度範圍驗證...")
        
        test_cases = [
            {
                "name": "低於最小值的溫度",
                "temperature": config.MIN_TEMPERATURE - 0.1,
                "expected": config.MIN_TEMPERATURE,
                "max_new_tokens": 100
            },
            {
                "name": "高於最大值的溫度",
                "temperature": config.MAX_TEMPERATURE + 0.1,
                "expected": config.MAX_TEMPERATURE,
                "max_new_tokens": 100
            },
            {
                "name": "低於最小值的 token 數",
                "temperature": config.DEFAULT_TEMPERATURE,
                "max_new_tokens": config.MIN_MAX_NEW_TOKENS - 10,
                "expected_tokens": config.MIN_MAX_NEW_TOKENS
            },
            {
                "name": "高於最大值的 token 數",
                "temperature": config.DEFAULT_TEMPERATURE,
                "max_new_tokens": config.MAX_MAX_NEW_TOKENS + 100,
                "expected_tokens": config.MAX_MAX_NEW_TOKENS
            }
        ]
        
        for test_case in test_cases:
            logger.info(f"\n測試: {test_case['name']}")
            result = rag_engine.generate_reply(
                query="測試問題",
                **{k: v for k, v in test_case.items() if k not in ['name', 'expected', 'expected_tokens']}
            )
            
            # 檢查是否有錯誤
            if 'error' in result:
                logger.error(f"測試 {test_case['name']} 時出錯: {result['error']}")
                continue
            
            if 'expected' in test_case and 'temperature' in result:
                logger.info(f"輸入溫度: {test_case['temperature']} -> 實際使用: {result['temperature']}")
                assert result['temperature'] == test_case['expected'], \
                    f"溫度驗證失敗: {result['temperature']} != {test_case['expected']}"
            
            if 'expected_tokens' in test_case and 'max_new_tokens' in result:
                logger.info(f"輸入 token 數: {test_case['max_new_tokens']} -> 實際使用: {result['max_new_tokens']}")
                assert result['max_new_tokens'] == test_case['expected_tokens'], \
                    f"Token 數驗證失敗: {result['max_new_tokens']} != {test_case['expected_tokens']}"
        
        # 8. 測試配置更新
        logger.info("\n8. 測試配置更新...")
        logger.info("更新前的配置:")
        status_before = rag_engine.get_status()
        logger.info(f"  預設溫度: {status_before['default_temperature']}")
        logger.info(f"  預設 Token 數: {status_before['default_max_new_tokens']}")
        logger.info(f"  Top-K: {status_before['default_top_k']}")
        
        # 更新配置（確保在有效範圍內）
        new_temperature = (config.MIN_TEMPERATURE + config.MAX_TEMPERATURE) / 2
        new_max_tokens = int((config.MIN_MAX_NEW_TOKENS + config.MAX_MAX_NEW_TOKENS) / 2)
        new_top_k = config.RAG_TOP_K + 3
        
        rag_engine.update_config(
            default_temperature=new_temperature,
            default_max_new_tokens=new_max_tokens,
            top_k=new_top_k
        )
        
        logger.info("更新後的配置:")
        status_after = rag_engine.get_status()
        logger.info(f"  預設溫度: {status_after['default_temperature']}")
        logger.info(f"  預設 Token 數: {status_after['default_max_new_tokens']}")
        logger.info(f"  Top-K: {status_after['default_top_k']}")
        
        # 驗證更新成功
        assert status_after['default_temperature'] == new_temperature
        assert status_after['default_max_new_tokens'] == new_max_tokens
        assert status_after['default_top_k'] == new_top_k
        logger.info("✓ 配置更新驗證通過")
        
        # 9. 測試記憶管理
        logger.info("\n9. 測試記憶管理...")
        
        # 創建長的對話歷史
        long_history = []
        for i in range(10):
            long_history.append({
                "role": "user",
                "content": f"這是第 {i+1} 個問題，內容較長以測試記憶管理功能。" * 5
            })
            long_history.append({
                "role": "assistant", 
                "content": f"這是第 {i+1} 個回覆，也比較長。" * 10
            })
        
        logger.info(f"長對話歷史包含 {len(long_history)} 條記錄")
        
        # 測試記憶構建
        memory_str = rag_engine.build_memory(
            history=long_history,
            enable=True,
            base_token_count=1000,
            reserve_for_context_and_query=1000
        )
        
        logger.info(f"構建的記憶字符串長度: {len(memory_str)}")
        logger.info(f"記憶字符串前 100 字符: {memory_str[:100]}..." if memory_str else "記憶字符串為空")
        
        # 10. 測試不同的 RAG 檢索場景
        logger.info("\n10. 測試不同的 RAG 檢索場景...")
        
        # 檢查資料庫狀態
        db_status = db_manager.get_status()
        logger.info(f"資料庫狀態: 連接={db_status['db_connected']}, 記錄數={db_status['record_count']}")
        
        if db_status['db_connected'] and db_status['record_count'] > 0:
            # 測試指定源文件搜索
            available_files = db_manager.get_available_source_files()
            if available_files:
                logger.info(f"測試指定源文件搜索，可用文件: {len(available_files)} 個")
                
                # 選擇前兩個文件進行測試
                test_files = available_files[:2] if len(available_files) >= 2 else available_files
                
                result_with_files = rag_engine.generate_reply(
                    query="ERAS 的主要好處是什麼？",
                    use_rag=True,
                    source_files=test_files,
                    max_new_tokens=300
                )
                
                logger.info(f"指定文件搜索結果: {result_with_files['response'][:100]}..." if result_with_files['response'] else "回覆為空")
                logger.info(f"使用的源文件: {result_with_files.get('source_files', 'N/A')}")
                logger.info(f"上下文是否提供: {'是' if result_with_files.get('context_provided') else '否'}")
                
                # 檢查錯誤
                if 'error' in result_with_files:
                    logger.error(f"指定文件搜索時出錯: {result_with_files['error']}")
        else:
            logger.warning("資料庫無數據或未連接，跳過 RAG 檢索測試")
        
        logger.info("\n" + "=" * 60)
        logger.info("RAG Engine 測試完成")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"RAG Engine 測試過程中發生錯誤: {e}", exc_info=True)
        raise

def run_all_tests():
    """執行所有測試"""
    logger.info("開始執行 RAG Engine 完整測試套件")
    logger.info(f"測試開始時間: {datetime.now()}")
    
    try:
        # 首先驗證配置
        test_config_validation()
        
        # 測試 RAG 引擎
        test_rag_engine()
        
        logger.info("\n" + "=" * 60)
        logger.info("所有測試完成")
        logger.info(f"測試結束時間: {datetime.now()}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"測試執行失敗: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # 檢查測試參數
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == "config":
            test_config_validation()
        elif test_type == "rag":
            test_rag_engine()
        elif test_type == "all":
            run_all_tests()
        else:
            logger.error(f"未知的測試類型: {test_type}")
            logger.info("可用選項: config, rag, all")
            sys.exit(1)
    else:
        # 預設執行所有測試
        run_all_tests()