#!/bin/bash

for j in {1..50..1};
do
  python svm.py --num=$j
done
