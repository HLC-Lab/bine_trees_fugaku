#!/bin/bash
DECOMPRESS=false

while getopts d flag
do
    case "${flag}" in
        d) DECOMPRESS=true;;
    esac
done

for folder in $(ls .)
do
    if [ -d "$folder" ]; then
        pushd $folder > /dev/null
        for dt in $(ls .)
        do
            if [ "$DECOMPRESS" = true ] ; then
                if [[ ${dt} == *.tar.xz ]]; then
                    tar -xvf ${dt}  
                fi
            else
                # Compress only folders
                if [ -d "$dt" ]; then         
                    tar vcfJ ${dt}.tar.xz ${dt}
                fi
            fi
        done
        popd > /dev/null
    fi
done