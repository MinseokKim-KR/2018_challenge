#!/bin/bash

for j in {1..50..1};
do
  python svm3.py --num=$j
  python svm4_1.py --num=$j
  python svm4_2.py --num=$j
  python svm5.py --num=$j
  python svm6.py --num=$j
done
