#!/usr/bin/bash

git_root=`git rev-parse --show-toplevel | sed -e 's/^C:/\/c/'`

pushd $git_root > /dev/null
git diff --stat=100,80 origin/master | grep -oe 'rtl_src/\w\+.v' > bcuda_files

rm -rf $git_root/../rx-fpga/design/src/rtl_src
mkdir -p $git_root/../rx-fpga/design/src/rtl_src
for f in `cat bcuda_files`; do
    cp -- "$f" "$git_root/../rx-fpga/design/src/$f"
done

cd $git_root/../rx-fpga/design/src/rtl_src > /dev/null
perl -p -i -e 's/bcuda_defines\.v/\.\.\/bcuda_defines\.v/' *.v
rm *.v.bak

popd
