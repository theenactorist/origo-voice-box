#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_app="${1:-/Applications/Origo Voice Box.app}"

if [[ "$(uname -s)" != Darwin ]]; then
  echo 'This local packaging script supports macOS.' >&2
  exit 1
fi

for tool in bun cargo rustc xcrun; do
  command -v "$tool" >/dev/null || { echo "Missing build tool: $tool" >&2; exit 1; }
done

plist="$source_app/Contents/Info.plist"
[[ -f "$plist" ]] || { echo "Application bundle not found: $source_app" >&2; exit 1; }
identifier="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$plist")"
version="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$plist")"
[[ "$identifier" == sh.voicebox.app && "$version" == 0.5.0 ]] || {
  echo 'The source bundle must be Voicebox / Origo Voice Box 0.5.0 (sh.voicebox.app).' >&2
  exit 1
}

target_triple="$(rustc -vV | awk '/^host:/ {print $2}')"
binaries="$repo_dir/tauri/src-tauri/binaries"
mkdir -p "$binaries"
for name in voicebox-server voicebox-mcp; do
  [[ -x "$source_app/Contents/MacOS/$name" ]] || { echo "Missing sidecar: $name" >&2; exit 1; }
  cp -p "$source_app/Contents/MacOS/$name" "$binaries/$name-$target_triple"
done

cd "$repo_dir"
bun install --frozen-lockfile --ignore-scripts
cd tauri
APPLE_SIGNING_IDENTITY=- bun run tauri build --bundles app -- --locked
