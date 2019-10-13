#!/bin/bash
bl_file=$1
remote_bl_file=~/tmp/$bl_file
# scp
rsync -ahvz $bl_file euler:~/tmp/
echo "File transfer finished"

# ssh render on remote
render_string="blender -b -noaudio ~/tmp/${bl_file} -o //${bl_file/.blend}_output.png -t 24 -f 1"  
echo $render_string
ssh euler "bsub -n 24 \"$render_string\""
echo "job submitted"

# rsync euler:~/tmp/output*.png ./
