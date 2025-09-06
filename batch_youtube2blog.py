#!/usr/bin/env python3
"""
YouTube動画のURLリストを順次処理してブログ記事を作成するバッチスクリプト
"""

import argparse
import os
import sys
import subprocess
import time
import logging
from typing import List, Dict
from pathlib import Path

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_youtube2blog.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def read_url_list(file_path: str) -> List[str]:
    """URLリストファイルを読み込む"""
    urls = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):  # 空行とコメント行をスキップ
                    urls.append(line)
        logger.info(f"URLリストファイルから {len(urls)} 個のURLを読み込みました")
        return urls
    except FileNotFoundError:
        logger.error(f"URLリストファイルが見つかりません: {file_path}")
        return []
    except Exception as e:
        logger.error(f"URLリストファイルの読み込みに失敗しました: {e}")
        return []

def run_youtube2blog_local(url: str, args: Dict) -> bool:
    """youtube2blog_local.pyを実行"""
    try:
        # コマンドライン引数を構築
        cmd = [
            sys.executable, 'youtube2blog_local.py',
            url
        ]
        
        # オプション引数を追加
        if args.get('whisper_model'):
            cmd.extend(['--whisper-model', args['whisper_model']])
        if args.get('output_language'):
            cmd.extend(['--output-language', args['output_language']])
        if args.get('min_words'):
            cmd.extend(['--min-words', str(args['min_words'])])
        if args.get('max_words'):
            cmd.extend(['--max-words', str(args['max_words'])])
        
        logger.info(f"実行コマンド: {' '.join(cmd)}")
        
        # サブプロセスで実行
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode == 0:
            logger.info(f"✅ URL処理成功: {url}")
            if result.stdout:
                logger.info(f"出力: {result.stdout.strip()}")
            return True
        else:
            logger.error(f"❌ URL処理失敗: {url}")
            if result.stderr:
                logger.error(f"エラー: {result.stderr.strip()}")
            return False
            
    except Exception as e:
        logger.error(f"❌ URL処理中に例外が発生: {url} - {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="YouTube動画のURLリストを順次処理してブログ記事を作成するバッチスクリプト"
    )
    parser.add_argument(
        "url_list_file",
        help="YouTube URLのリストファイル（1行に1つのURL）"
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="使用するWhisperモデル (デフォルト: base)"
    )
    parser.add_argument(
        "--output-language",
        default="ja",
        choices=["ja", "zh"],
        help="出力するブログ記事の言語 (ja または zh)"
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=3000,
        help="ブログ記事の最小目標文字数 (デフォルト: 2000)"
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=3500,
        help="ブログ記事の最大目標文字数 (デフォルト: 2500)"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="URL処理間の待機時間（秒） (デフォルト: 5)"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="エラーが発生しても次のURLの処理を続行する"
    )
    
    args = parser.parse_args()
    
    # URLリストファイルの存在確認
    if not os.path.exists(args.url_list_file):
        logger.error(f"URLリストファイルが見つかりません: {args.url_list_file}")
        sys.exit(1)
    
    # youtube2blog_local.pyの存在確認
    if not os.path.exists('youtube2blog_local.py'):
        logger.error("youtube2blog_local.pyが見つかりません。同じディレクトリに配置してください。")
        sys.exit(1)
    
    # URLリストを読み込み
    urls = read_url_list(args.url_list_file)
    if not urls:
        logger.error("処理するURLがありません。")
        sys.exit(1)
    
    # 実行オプションを辞書に格納
    execution_args = {
        'whisper_model': args.whisper_model,
        'output_language': args.output_language,
        'min_words': args.min_words,
        'max_words': args.max_words
    }
    
    logger.info("=== バッチ処理開始 ===")
    logger.info(f"処理対象URL数: {len(urls)}")
    logger.info(f"Whisperモデル: {args.whisper_model}")
    logger.info(f"出力言語: {args.output_language}")
    logger.info(f"文字数範囲: {args.min_words}〜{args.max_words}")
    logger.info(f"処理間待機時間: {args.delay}秒")
    logger.info(f"エラー時継続: {args.continue_on_error}")
    
    # 統計情報
    success_count = 0
    failure_count = 0
    
    # URLを順次処理
    for i, url in enumerate(urls, 1):
        logger.info(f"\n--- 処理 {i}/{len(urls)} ---")
        logger.info(f"URL: {url}")
        
        start_time = time.time()
        success = run_youtube2blog_local(url, execution_args)
        elapsed_time = time.time() - start_time
        
        if success:
            success_count += 1
            logger.info(f"処理時間: {elapsed_time:.2f}秒")
        else:
            failure_count += 1
            logger.error(f"処理時間: {elapsed_time:.2f}秒")
            
            if not args.continue_on_error:
                logger.error("エラーが発生したため処理を中断します。")
                break
        
        # 最後のURLでなければ待機
        if i < len(urls):
            logger.info(f"{args.delay}秒待機中...")
            time.sleep(args.delay)
    
    # 結果サマリー
    logger.info("\n=== バッチ処理完了 ===")
    logger.info(f"成功: {success_count}件")
    logger.info(f"失敗: {failure_count}件")
    logger.info(f"成功率: {success_count/(success_count+failure_count)*100:.1f}%")
    
    if failure_count > 0 and not args.continue_on_error:
        sys.exit(1)

if __name__ == "__main__":
    main()
