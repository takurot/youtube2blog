#!/bin/bash
# YouTube動画からローカルで文字起こしを生成してブログ記事を作成するスクリプト

# 使用方法を表示
show_usage() {
    echo "使用方法: $0 <youtube_url> [whisper_model] [options]"
    echo ""
    echo "例:"
    echo "  $0 'https://www.youtube.com/watch?v=example'"
    echo "  $0 'https://www.youtube.com/watch?v=example' medium"
    echo "  $0 'https://www.youtube.com/watch?v=example' base --blog-only"
    echo ""
    echo "Whisperモデル: tiny, base, small, medium, large (デフォルト: base)"
    echo "オプション:"
    echo "  --blog-only     ブログ記事のみを生成"
    echo "  --min-words N   最小文字数 (デフォルト: 2500)"
    echo "  --max-words N   最大文字数 (デフォルト: 3000)"
    echo ""
}

# 引数チェック
if [ $# -eq 0 ]; then
    echo "エラー: YouTube URLが指定されていません。"
    show_usage
    exit 1
fi

# 必要なソフトウェアのチェック
check_dependencies() {
    echo "依存関係をチェック中..."
    
    # Python3の確認
    if ! command -v python3 &> /dev/null; then
        echo "エラー: Python3がインストールされていません。"
        exit 1
    fi
    
    # FFmpegの確認
    if ! command -v ffmpeg &> /dev/null; then
        echo "エラー: FFmpegがインストールされていません。"
        echo "macOSの場合: brew install ffmpeg"
        echo "Ubuntuの場合: sudo apt install ffmpeg"
        exit 1
    fi
    
    echo "依存関係チェック完了。"
}

# 仮想環境の設定とパッケージインストール
setup_environment() {
    echo "環境を準備中..."
    
    # 仮想環境が存在しない場合は作成
    if [ ! -d ".env" ]; then
        echo "仮想環境を作成中..."
        python3 -m venv .env
    fi
    
    # 仮想環境をアクティベート
    source .env/bin/activate
    
    # パッケージをインストール
    echo "必要なパッケージをインストール中..."
    pip install -r requirements.txt
    
    echo "環境準備完了。"
}

# メイン処理
main() {
    local youtube_url="$1"
    local whisper_model="${2:-base}"
    # 3番目以降を追加オプションとして扱う（shiftは使わない）
    local additional_args="${@:3}"
    
    echo "=== YouTube動画からローカル文字起こし生成 ==="
    echo "動画URL: $youtube_url"
    echo "Whisperモデル: $whisper_model"
    echo "追加オプション: $additional_args"
    echo ""
    
    # 依存関係チェック
    check_dependencies
    
    # 環境設定
    setup_environment
    
    # 仮想環境をアクティベート
    source .env/bin/activate
    
    # Python スクリプトを実行
    echo "Python スクリプトを実行中..."
    python3 youtube2blog_local.py "$youtube_url" --whisper-model "$whisper_model" $additional_args
    
    # 実行結果を表示
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 処理が正常に完了しました。"
        echo "生成されたファイルを確認してください。"
    else
        echo ""
        echo "❌ 処理中にエラーが発生しました。"
        exit 1
    fi
}

# ヘルプオプション
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

# メイン処理を実行
main "$@" 