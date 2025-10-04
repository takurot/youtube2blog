#!/usr/bin/env bash
set -euo pipefail

print_usage() {
  cat << 'USAGE'
使い方:
  extract_popular_top.sh <CHANNEL_URL> [-n COUNT] [-o OUTPUT]

例:
  extract_popular_top.sh "https://www.youtube.com/@EnesYilmazer/videos" -n 50 > top50_urls.txt
  extract_popular_top.sh "https://www.youtube.com/@EnesYilmazer" -n 50 -o top50_urls.txt

オプション:
  -n, --count COUNT   取得件数 (デフォルト: 50)
  -o, --output FILE   出力先ファイル (指定しない場合は標準出力)
  -h, --help          このヘルプを表示
USAGE
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "エラー: コマンド '$1' が見つかりません。インストールしてください。" >&2
    exit 1
  fi
}

normalize_channel_url() {
  # 入力: $1 (チャンネルURL: /videos の有無は任意)
  # 出力: 人気順クエリ付きURL
  local input_url="$1"

  # /videos を付与
  local base
  if [[ "$input_url" == *"/videos"* ]]; then
    base="$input_url"
  else
    base="${input_url%/}/videos"
  fi

  # 既存クエリを除去して人気順用クエリを付与
  local base_no_q="${base%%\?*}"
  echo "${base_no_q}?view=0&sort=p&flow=grid"
}

main() {
  local count=50
  local output=""

  # 引数が無い場合
  if [[ $# -eq 0 ]]; then
    print_usage
    exit 1
  fi

  local channel_url=""

  # 最初の位置引数をURLとして受け取り、残りはオプション解析
  channel_url="$1"; shift || true

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -n|--count)
        shift
        [[ $# -gt 0 ]] || { echo "エラー: -n/--count の値がありません" >&2; exit 1; }
        count="$1"
        ;;
      -o|--output)
        shift
        [[ $# -gt 0 ]] || { echo "エラー: -o/--output の値がありません" >&2; exit 1; }
        output="$1"
        ;;
      -h|--help)
        print_usage
        exit 0
        ;;
      *)
        echo "不明な引数: $1" >&2
        print_usage
        exit 1
        ;;
    esac
    shift || true
  done

  # 必要コマンド確認
  require_cmd yt-dlp
  require_cmd sed
  require_cmd awk

  # URL整形
  final_url="$(normalize_channel_url "$channel_url")"

  # 取得コマンド
  # -I 1:${count}: 上位N件
  # --flat-playlist: URLのみ高速取得
  # awk '!seen[$0]++': 重複除去(順序保持)
  if [[ -n "$output" ]]; then
    yt-dlp --flat-playlist -I "1:${count}" --print "%(webpage_url)s" "$final_url" \
      | sed '/^$/d' \
      | awk '!seen[$0]++' \
      | tee "$output" >/dev/null
  else
    yt-dlp --flat-playlist -I "1:${count}" --print "%(webpage_url)s" "$final_url" \
      | sed '/^$/d' \
      | awk '!seen[$0]++'
  fi
}

main "$@"


