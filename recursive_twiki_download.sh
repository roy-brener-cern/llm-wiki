#!/bin/bash
# recursive_twiki_download.sh
# Usage: ./recursive_twiki_download.sh cookies.json output_folder

COOKIES=$1
OUTDIR=$2

mkdir -p "$OUTDIR"
QUEUE=("WebHome")       # start with WebHome
SEEN=()

while [ ${#QUEUE[@]} -gt 0 ]; do
    PAGE=${QUEUE[0]}
    QUEUE=("${QUEUE[@]:1}")

    # Skip if already downloaded
    if [[ " ${SEEN[@]} " =~ " ${PAGE} " ]]; then
        continue
    fi

    echo "Downloading $PAGE ..."
    # Save raw content
    curl -s -b "$COOKIES" -L "https://twiki.cern.ch/twiki/bin/view/AtlasProtected/$PAGE?raw=on" \
        -o "$OUTDIR/$PAGE.txt"

    # Fetch immediate children
    CHILD_HTML=$(curl -s -b "$COOKIES" -L "https://twiki.cern.ch/twiki/bin/view/AtlasProtected/$PAGE?topicparent=$PAGE&skin=print")
    CHILD_PAGES=($(echo "$CHILD_HTML" | grep -o 'href="/twiki/bin/view/AtlasProtected/[^"]*"' \
                   | sed 's|href="/twiki/bin/view/AtlasProtected/||' | sed 's|"||'))

    # Add new children to queue
    for C in "${CHILD_PAGES[@]}"; do
        if [[ ! " ${SEEN[@]} " =~ " ${C} " ]]; then
            QUEUE+=("$C")
        fi
    done

    SEEN+=("$PAGE")
done

echo "Done! All pages saved in $OUTDIR"

