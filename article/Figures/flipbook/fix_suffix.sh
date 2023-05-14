#!/bin/bash
# Remove leading zeroes withn in the name

for file in ./*.png; do
    NUM=$(echo "$file$" | tr -dc '0-9')
    NONZERO=$(echo $NUM | sed 's/^0*//')
    NEW="flip"$NONZERO".png"
    echo $NEW
    mv $file $NEW
done