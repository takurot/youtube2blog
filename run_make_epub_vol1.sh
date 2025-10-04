#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT="$SCRIPT_DIR"
ENV_PY="$PROJECT_ROOT/.env/bin/python3"
VENV_PY="$PROJECT_ROOT/.venv/bin/python3"
if [ -x "$ENV_PY" ]; then
  PYTHON="$ENV_PY"
elif [ -x "$VENV_PY" ]; then
  PYTHON="$VENV_PY"
else
  PYTHON="python3"
fi

INPUT_DIR="/Volumes/SSD2TB_DATA/src/youtube2blog/millionaires/backup/vol1"
OUTPUT_DIR="$PROJECT_ROOT/output"
OUTPUT_PATH="$OUTPUT_DIR/millionaires_vol1.epub"

TITLE="億万長者のリアル Vol.1"
AUTHOR="youtube2blog 編集部"
LANG="ja"
DESCRIPTION="ウォール街で出会った成功者へのインタビュー要点を精選。地味で堅実なキャッシュフロービジネス、目的意識、ネットワークの力などを体系化。"
COVER=""

mkdir -p "$OUTPUT_DIR"

$PYTHON "$PROJECT_ROOT/make_epub_from_dir.py" "$INPUT_DIR" \
  --title "$TITLE" \
  --author "$AUTHOR" \
  --lang "$LANG" \
  --description "$DESCRIPTION" \
  ${COVER:+--cover "$COVER"} \
  --output "$OUTPUT_PATH"

echo "Created: $OUTPUT_PATH"


