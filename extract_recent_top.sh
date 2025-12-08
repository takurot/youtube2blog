#!/usr/bin/env bash
set -euo pipefail

print_usage() {
  cat << 'USAGE'
使い方:
  extract_recent_top.sh <CHANNEL_URL> [-n COUNT] [-o OUTPUT]

例:
  extract_recent_top.sh "https://www.youtube.com/@EnesYilmazer/videos" -n 50 > recent50_urls.txt
  extract_recent_top.sh "https://www.youtube.com/@EnesYilmazer" -n 50 -o recent50_urls.txt

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
  require_cmd head

  # URL整形（/videos を付与、クエリは付けない）
  local base
  if [[ "$channel_url" == *"/videos"* ]]; then
    base="${channel_url%%\?*}"
  else
    base="${channel_url%/}/videos"
  fi
  
  final_url="${base%%\?*}"

  # 取得コマンド
  # --flat-playlist: メタデータのみ高速取得
  # デフォルトで新しい順（時系列順）に取得されるため、そのまま順序を保持
  # awk '!seen[$0]++': 重複除去(順序保持)
  if [[ -n "$output" ]]; then
    yt-dlp --flat-playlist --print "%(webpage_url)s" "$final_url" \
      | sed '/^$/d' \
      | head -n "${count}" \
      | awk '!seen[$0]++' \
      | tee "$output" >/dev/null
  else
    yt-dlp --flat-playlist --print "%(webpage_url)s" "$final_url" \
      | sed '/^$/d' \
      | head -n "${count}" \
      | awk '!seen[$0]++'
  fi
}

main "$@"




