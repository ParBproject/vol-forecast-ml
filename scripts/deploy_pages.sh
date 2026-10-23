#!/usr/bin/env bash
# Publish web/dist to the gh-pages branch (GitHub Pages serves it).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
cp -r "$ROOT/web/dist/." "$TMP/"
touch "$TMP/.nojekyll"
cd "$TMP"
git init -q -b gh-pages
git add -A
git -c user.name="Par Bahrae" -c user.email="225519193+ParBproject@users.noreply.github.com" commit -q -m "Deploy $(git -C "$ROOT" rev-parse --short HEAD)"
git push -q -f "$(git -C "$ROOT" remote get-url origin)" gh-pages
rm -rf "$TMP"
