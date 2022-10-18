#!/usr/bin/env bash

# IN="bla@some.com;john@home.com"

# mails=$(echo $IN | tr ";" "\n")


for file in *.restart; do    
    [ -f "$file" ] || break    
    name="${file%.*}"_folder
    
    mkdir "$name"
    mv "$file" "$name"/"$file" # put mv
    echo "put some text in here" > "$name"/job.sh

done

