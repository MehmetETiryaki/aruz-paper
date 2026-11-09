#!/bin/bash
conda activate aruz
cd "$(dirname "$0")/.."
echo "host $(hostname), label $1, provider $2, model $3, split $4, args ${@:5}"
for v in ANTHROPIC_API_KEY OPENAI_API_KEY GEMINI_API_KEY OPENROUTER_API_KEY; do echo "$v: $(eval echo \${#$v}) chars"; done
exec scripts/llm_run_resumable.sh "$@"
