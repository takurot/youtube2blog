#!/bin/bash

rm ./._*
rm ./202*.*

# youtube2blog.sh - YouTube の video ID を引数として youtube2blog.py を実行するスクリプト

# 使用方法の表示関数
show_usage() {
    echo "使用方法: $0 VIDEO_ID [LANGUAGE] [CHANNEL] [OPTIONS]"
    echo ""
    echo "引数:"
    echo "  VIDEO_ID     YouTube の video ID (例: 5uBAQrg4SoQ)"
    echo "  LANGUAGE     字幕の言語コード (デフォルト: ja)"
    echo "  CHANNEL      YouTubeアップロード用のチャンネル名 (省略可能)"
    echo ""
    echo "オプション:"
    echo "  --with-media ブログ記事だけでなく、音声・動画も生成する"
    echo ""
    echo "例:"
    echo "  $0 5uBAQrg4SoQ"
    echo "  $0 5uBAQrg4SoQ en"
    echo "  $0 5uBAQrg4SoQ en channel1"
    echo "  $0 5uBAQrg4SoQ en --with-media"
    echo "  $0 5uBAQrg4SoQ en channel1 --with-media"
    exit 1
}

# 引数のチェック
if [ $# -lt 1 ]; then
    show_usage
fi

# 第1引数は必須 (VIDEO_ID)
VIDEO_ID=$1
shift

# 第2引数は任意 (LANGUAGE)
LANGUAGE="ja"
if [ $# -gt 0 ] && [[ $1 != --* ]]; then
    LANGUAGE=$1
    shift
fi

# 第3引数は任意 (CHANNEL)
CHANNEL=""
if [ $# -gt 0 ] && [[ $1 != --* ]]; then
    CHANNEL=$1
    shift
fi

# デフォルトでブログ記事のみのモードを有効化
BLOG_ONLY=true
EXTRA_ARGS=()

# オプションの処理
while [ $# -gt 0 ]; do
    case "$1" in
        --with-media)
            BLOG_ONLY=false
            ;;
        *)
            # ここで他の youtube2blog.py 用オプションをハンドルする場合
            # 例: --shorts, --no-bgm など
            # EXTRA_ARGS="$EXTRA_ARGS $1"
            # 今回はシンプルにするため、他のオプションは無視
            ;;
    esac
    shift
done

# モードに応じて引数を設定
if [ "$BLOG_ONLY" = true ]; then
    EXTRA_ARGS+=(--blog-only)
    EXTRA_ARGS+=(--min-words 2500 --max-words 3000) # ブログモードは長めに
else
    # メディア生成モードの場合のデフォルトオプション
    EXTRA_ARGS+=(--wordcloud)
    EXTRA_ARGS+=(--min-words 2000 --max-words 2500) # メディア生成でも記事は長めに
    # ここで --no-bgm や --shorts などのオプションを追加することも可能
    # 例: if [ <no-bgmが指定されたか> ]; then EXTRA_ARGS="$EXTRA_ARGS --no-bgm"; fi
fi

# YouTube URL の構築
YOUTUBE_URL="https://www.youtube.com/watch?v=$VIDEO_ID"

# コマンドの実行
# echo "YouTube URL: $YOUTUBE_URL"
# echo "言語コード: $LANGUAGE"

if [ "$BLOG_ONLY" = true ]; then
    echo "ブログ記事のみ生成モードで実行します..."
else
    echo "音声・動画も含めて生成します..."
fi

# ブログ記事の生成
echo "ブログ記事の生成を開始します..."
python youtube2blog.py "$LANGUAGE" "$YOUTUBE_URL" "${EXTRA_ARGS[@]}"

# ブログ記事のみの場合は終了
if [ "$BLOG_ONLY" = true ]; then
    echo "ブログ記事のみを生成しました。処理を終了します。"
    # exit 0
fi

# exit 0 