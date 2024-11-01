#!/bin/bash

# for file in *; 表示遍历所有文件或文件夹(不是recursively)，';'用来隔开同一行的两个命令，do也可以放在下一行，不用缩进
for file in * .*; do
  # Ensure we don't rename the current or parent directory
  if [ "$file" != "." ] && [ "$file" != ".." ]; then
    mv "$file" "abc$file"
  fi   # ends the if block
done
# 'done' mark the end of for loop


# 一般不管隐藏文件，直接用下面这个重命名，所有文件或文件夹名前加个'abc'
for file in *
do
  mv "$file" "abc$file"
done