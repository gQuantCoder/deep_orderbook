#! /bin/bash


set -e

clear

pytest \
-n=16 \
 $1 $2 \
--cov-report xml:cov.xml \
--cov deep_orderbook \
--durations=20 \
--timeout=100 \
--failed-first
