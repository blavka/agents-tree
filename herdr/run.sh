#!/usr/bin/env bash
# Run agents-tree's herdr glue from the copy bundled with the plugin: no install needed,
# only a python3 that is 3.10 or newer (or AGENTS_TREE_PYTHON pointing at one).
set -euo pipefail

root="${HERDR_PLUGIN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
py="${AGENTS_TREE_PYTHON:-python3}"

if ! "$py" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
  echo "agents-tree needs Python 3.10+ as '$py' (set AGENTS_TREE_PYTHON to use another)." >&2
  if [ "${1:-}" = pane ]; then read -r -p "  press enter to close " _ || true; fi
  exit 1
fi

PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}" exec "$py" -m agents_tree.herdr "$@"
