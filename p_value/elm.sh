#!/bin/bash

for j in {1..50..1};
do
  python elm.py --num=$j
  python elm3.py --num=$j
  python elm4_1.py --num=$j
  python elm4_2.py --num=$j
  python elm5.py --num=$j
  python elm6.py --num=$j

done
