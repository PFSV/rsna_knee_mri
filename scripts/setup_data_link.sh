#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
data_root=${1:-${RSNA_DATA_ROOT:-}}
link="$repo_root/data"

if [ -z "$data_root" ]; then
    printf 'Usage: %s /path/to/rsna/data\n' "$0" >&2
    printf 'Or set RSNA_DATA_ROOT in the environment.\n' >&2
    exit 2
fi

for required in train.csv test.csv train_series.csv test_series.csv train_series test_series; do
    if [ ! -e "$data_root/$required" ]; then
        printf 'Missing required dataset path: %s\n' "$data_root/$required" >&2
        exit 1
    fi
done

if [ -L "$link" ]; then
    current=$(readlink "$link")
    if [ "$current" != "$data_root" ]; then
        printf 'Existing data link points elsewhere: %s -> %s\n' "$link" "$current" >&2
        exit 1
    fi
elif [ -e "$link" ]; then
    printf 'Refusing to replace existing non-symlink path: %s\n' "$link" >&2
    exit 1
else
    ln -s "$data_root" "$link"
fi

train_studies=$(find "$data_root/train_series" -mindepth 1 -maxdepth 1 -type d | wc -l)
test_studies=$(find "$data_root/test_series" -mindepth 1 -maxdepth 1 -type d | wc -l)

printf 'RSNA data ready\n'
printf '  repository: %s\n' "$repo_root"
printf '  data link:  %s -> %s\n' "$link" "$data_root"
printf '  studies:    train=%s test=%s\n' "$train_studies" "$test_studies"
