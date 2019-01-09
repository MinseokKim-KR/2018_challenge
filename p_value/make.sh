#!/bin/bash

Test = ['test1', 'test2', 'test3']
Driver = ['TW','JH','HH','CS']
Dates = ['170217', '170220', '170518', '170530', '170601']
Name = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11']

for dr in 0 1 2 3;
do
  for da in 0 1 2 3 4;
  do
    for na in 0 1 2 3 4 5 6 7 8 9 10;
    do
      for te in 0 1 2;
      do
        print "%s" Driver[$dr]
        python making_input5.py --Driver=Driver[$dr] --Dates=Dates[$da] --Name=Name[$na] --Test=Test[$te]
      done
    done
  done
done