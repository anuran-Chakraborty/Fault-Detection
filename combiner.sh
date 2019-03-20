#!/bin/bash

cat csv_train/*.csv >single_train.csv
cat csv_test/*.csv >single_test.csv

cat multi_csv_train/*.csv >multi_train.csv
cat multi_csv_test/*.csv >multi_test.csv

cat single_train.csv multi_train.csv >train.csv
cat single_test.csv multi_test.csv >test.csv
