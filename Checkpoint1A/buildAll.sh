#!/bin/bash

## Script for ECEN 5593 Summer 2016, Checkpoint-1A
#  Diana Southard
#
# Create script for each inidividual benchmark to run, as set up by using bench-run.pl
# 
# Start building scripts to run benchmarks

echo "Starting all benchmark script builds"

#bzip2
bench-run.pl --bench spec-cpu2006:int:401.bzip2:train --build base --prefix "pin -t /home/dianasouthard/Documents/checkpoint-1A/obj-intel64/checkpoint1A.so -o bzip2_1bit.out -l 10000000000 --" --copy bzip2_1bit.out --log LOG
mv go.sh run_bzip_1bit.sh

bench-run.pl --bench spec-cpu2006:int:401.bzip2:train --build base --prefix "pin -t /home/dianasouthard/Documents/checkpoint-1A/obj-intel64/checkpoint1A.so -o bzip2_2bit.out -l 10000000000 -b 1 --" --copy bzip2_2bit.out --log LOG
mv go.sh run_bzip_2bit.sh

bench-run.pl --bench spec-cpu2006:int:401.bzip2:train --build base --prefix "pin -t /home/dianasouthard/Documents/checkpoint-1A/obj-intel64/checkpoint1A.so -o bzip2_gag.out -l 10000000000 -b 2 --" --copy bzip2_gag.out --log LOG
mv go.sh run_bzip_gag.sh

bench-run.pl --bench spec-cpu2006:int:401.bzip2:train --build base --prefix "pin -t /home/dianasouthard/Documents/checkpoint-1A/obj-intel64/checkpoint1A.so -o bzip2_pag.out -l 10000000000 -b 3 --" --copy bzip2_pag.out --log LOG
mv go.sh run_bzip_pag.sh

bench-run.pl --bench spec-cpu2006:int:401.bzip2:train --build base --prefix "pin -t /home/dianasouthard/Documents/checkpoint-1A/obj-intel64/checkpoint1A.so -o bzip2_hybrid.out -l 10000000000 -b 4 --" --copy bzip2_hybrid.out --log LOG
mv go.sh run_bzip_hybrid.sh


#sjeng
bench-run.pl --bench spec-cpu2006:int:458.sjeng:train --build base --prefix "pin -t /home/dianasouthard/Documents/checkpoint-1A/obj-intel64/checkpoint1A.so -o sjeng_1bit.out -l 10000000000 --" --copy sjeng_1bit.out --log LOG
mv go.sh run_sjeng_1bit.sh

bench-run.pl --bench spec-cpu2006:int:458.sjeng:train --build base --prefix "pin -t /home/dianasouthard/Documents/checkpoint-1A/obj-intel64/checkpoint1A.so -o sjeng_2bit.out -l 10000000000 -b 1 --" --copy sjeng_2bit.out --log LOG
mv go.sh run_sjeng_2bit.sh

bench-run.pl --bench spec-cpu2006:int:458.sjeng:train --build base --prefix "pin -t /home/dianasouthard/Documents/checkpoint-1A/obj-intel64/checkpoint1A.so -o sjeng_gag.out -l 10000000000 -b 2 --" --copy sjeng_gag.out --log LOG
mv go.sh run_sjeng_gag.sh

bench-run.pl --bench spec-cpu2006:int:458.sjeng:train --build base --prefix "pin -t /home/dianasouthard/Documents/checkpoint-1A/obj-intel64/checkpoint1A.so -o sjeng_pag.out -l 10000000000 -b 3 --" --copy sjeng_pag.out --log LOG
mv go.sh run_sjeng_pag.sh

bench-run.pl --bench spec-cpu2006:int:458.sjeng:train --build base --prefix "pin -t /home/dianasouthard/Documents/checkpoint-1A/obj-intel64/checkpoint1A.so -o sjeng_hybrid.out -l 10000000000 -b 4 --" --copy sjeng_hybrid.out --log LOG
mv go.sh run_sjeng_hybrid.sh


#libquantum
bench-run.pl --bench spec-cpu2006:int:462.libquantum:train --build base --prefix "pin -t /home/dianasouthard/Documents/checkpoint-1A/obj-intel64/checkpoint1A.so -o libquantum_1bit.out -l 10000000000 --" --copy libquantum_1bit.out --log LOG
mv go.sh run_libquantum_1bit.sh

bench-run.pl --bench spec-cpu2006:int:462.libquantum:train --build base --prefix "pin -t /home/dianasouthard/Documents/checkpoint-1A/obj-intel64/checkpoint1A.so -o libquantum_2bit.out -l 10000000000 -b 1 --" --copy libquantum_2bit.out --log LOG
mv go.sh run_libquantum_2bit.sh

bench-run.pl --bench spec-cpu2006:int:462.libquantum:train --build base --prefix "pin -t /home/dianasouthard/Documents/checkpoint-1A/obj-intel64/checkpoint1A.so -o libquantum_gag.out -l 10000000000 -b 2 --" --copy libquantum_gag.out --log LOG
mv go.sh run_libquantum_gag.sh

bench-run.pl --bench spec-cpu2006:int:462.libquantum:train --build base --prefix "pin -t /home/dianasouthard/Documents/checkpoint-1A/obj-intel64/checkpoint1A.so -o libquantum_pag.out -l 10000000000 -b 3 --" --copy libquantum_pag.out --log LOG
mv go.sh run_libquantum_pag.sh

bench-run.pl --bench spec-cpu2006:int:462.libquantum:train --build base --prefix "pin -t /home/dianasouthard/Documents/checkpoint-1A/obj-intel64/checkpoint1A.so -o libquantum_hybrid.out -l 10000000000 -b 4 --" --copy libquantum_hybrid.out --log LOG
mv go.sh run_libquantum_hybrid.sh


#h264
bench-run.pl --bench spec-cpu2006:int:464.h264ref:train --build base --prefix "pin -t /home/dianasouthard/Documents/checkpoint-1A/obj-intel64/checkpoint1A.so -o h264_1bit.out -l 10000000000 --" --copy h264_1bit.out --log LOG
mv go.sh run_h264_1bit.sh

bench-run.pl --bench spec-cpu2006:int:464.h264ref:train --build base --prefix "pin -t /home/dianasouthard/Documents/checkpoint-1A/obj-intel64/checkpoint1A.so -o h264_2bit.out -l 10000000000 -b 1 --" --copy h264_2bit.out --log LOG
mv go.sh run_h264_2bit.sh

bench-run.pl --bench spec-cpu2006:int:464.h264ref:train --build base --prefix "pin -t /home/dianasouthard/Documents/checkpoint-1A/obj-intel64/checkpoint1A.so -o h264_gag.out -l 10000000000 -b 2 --" --copy h264_gag.out --log LOG
mv go.sh run_h264_gag.sh

bench-run.pl --bench spec-cpu2006:int:464.h264ref:train --build base --prefix "pin -t /home/dianasouthard/Documents/checkpoint-1A/obj-intel64/checkpoint1A.so -o h264_pag.out -l 10000000000 -b 3 --" --copy h264_pag.out --log LOG
mv go.sh run_h264_pag.sh

bench-run.pl --bench spec-cpu2006:int:464.h264ref:train --build base --prefix "pin -t /home/dianasouthard/Documents/checkpoint-1A/obj-intel64/checkpoint1A.so -o h264_hybrid.out -l 10000000000 -b 4 --" --copy h264_hybrid.out --log LOG
mv go.sh run_h264_hybrid.sh


###################################### Benchmark Scripts Ready to Run!! #######################################3

echo "All scripts built. Starting to run benchmarks."
echo "Current line limit: 10000000000"

echo "Starting to run bzip"
sh ./run_bzip_1bit.sh
echo "    Finished 1 bit"
sh ./run_bzip_2bit.sh
echo "    Finished 2 bit"
sh ./run_bzip_gag.sh
echo "    Finished GAg"
sh ./run_bzip_pag.sh
echo "    Finished PAg"
sh ./run_bzip_hybrid.sh

echo "bzip completed. Starting sjeng"
sh ./run_sjeng_1bit.sh
echo "    Finished 1 bit"
sh ./run_sjeng_2bit.sh
echo "    Finished 2 bit"
sh ./run_sjeng_gag.sh
echo "    Finished GAg"
sh ./run_sjeng_pag.sh
echo "    Finished PAg"
sh ./run_sjeng_hybrid.sh

echo "sjeng completed. Starting libquantum"
sh ./run_libquantum_1bit.sh
echo "    Finished 1 bit"
sh ./run_libquantum_2bit.sh
echo "    Finished 2 bit"
sh ./run_libquantum_gag.sh
echo "    Finished GAg"
sh ./run_libquantum_pag.sh
echo "    Finished PAg"
sh ./run_libquantum_hybrid.sh

echo "libquantum completed. Starting h264"
sh ./run_h264_1bit.sh
echo "    Finished 1 bit"
sh ./run_h264_2bit.sh
echo "    Finished 2 bit"
sh ./run_h264_gag.sh
echo "    Finished GAg"
sh ./run_h264_pag.sh
echo "    Finished PAg"
sh ./run_h264_hybrid.sh

echo "Benchmarks completed."
echo "Removing scripts"

rm run_*.sh

## EOF ##
