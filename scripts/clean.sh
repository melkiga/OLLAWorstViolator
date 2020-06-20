#!/bin/bash

current=`pwd`
current_dir=`basename "$current"`
if [ "$current_dir" == "scripts" ]; then
    cd ..
fi

rm -rf build
rm -rf bin
rm -rf googletest