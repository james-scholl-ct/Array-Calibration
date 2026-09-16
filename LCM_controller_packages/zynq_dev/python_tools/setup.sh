#!/usr/bin/bash

# git_root=`git rev-parse --show-toplevel`
git_root=`git rev-parse --show-toplevel | sed -e 's/^C:/\/c/'`

path=$git_root
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=$path
else
    #export PYTHONPATH=$path:$PYTHONPATH
    export PYTHONPATH=$path
fi
echo "\$PYTHONPATH = $PYTHONPATH"
