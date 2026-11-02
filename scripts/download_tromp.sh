#!/bin/sh
# Fetches John Tromp's 8-ply Connect-4 database from the UCI repository into data/tromp/.
# Licence: CC BY 4.0, https://doi.org/10.24432/C59P43. The data is not committed here apart
# from the 60-row sample in tests/data.
set -eu
url="https://archive.ics.uci.edu/static/public/26/connect+4.zip"
expected="14c886b817783ac9b642f42ec78a2e67f727a3d41c94913af8954d454eee2367"
mkdir -p data/tromp
curl -fsSL "$url" -o data/tromp/connect-4.zip
unzip -o -q data/tromp/connect-4.zip -d data/tromp
if command -v sha256sum >/dev/null; then
    actual=$(sha256sum data/tromp/connect-4.data.Z | cut -d' ' -f1)
else
    actual=$(shasum -a 256 data/tromp/connect-4.data.Z | cut -d' ' -f1)
fi
if [ "$actual" != "$expected" ]; then
    echo "checksum mismatch: $actual" >&2
    exit 1
fi
gzip -dc data/tromp/connect-4.data.Z > data/tromp/connect-4.data
wc -l data/tromp/connect-4.data
