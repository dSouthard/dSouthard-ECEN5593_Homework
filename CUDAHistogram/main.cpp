/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
 /*
 * This sample implements 64-bin histogram calculation
 * of arbitrary-sized 8-bit data array
 */

// Utility and system includes
#include <shrUtils.h>
#include <shrQATest.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include<string>

// project include
#include "histogram_common.h"

#ifdef __DEVICE_EMULATION__
const int numRuns = 1;
#else
const int numRuns = 1;
#endif

static char *sSDKsample = "[histogram]\0";
bool atomicAddFlag = false;
uint binCount = 64;
//std::string fileName = "histogram";

int main(int argc, char **argv)
{
    uchar *h_Data;
    uint  *h_HistogramCPU, *h_HistogramGPU;
    uchar *d_Data;
    uint  *d_Histogram;
    uint hTimer = 0, transferTimer = 0;
    int PassFailFlag = 1;
    uint byteCount = 1000*4;
    uint uiSizeMult = 1;

    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev;

   std::string fileName = "histogram";
	shrQAStart(argc, argv);
    // Set input length
    for (int i = 1; i < argc; i++) {
	if (strcmp(argv[i], "-length") == 0){
		byteCount = atoi(argv[i+1]);
		fileName += argv[i+1];
	}
	else if (strcmp(argv[i], "-a") == 0) {
		atomicAddFlag = true;
		fileName += "A";
	}
	i++;
    }

	fileName += ".txt";
	// set logfile name and start logs
   	const char * c = fileName.c_str();
	 shrSetLogFileName (c);

    //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if( shrCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
        dev = cutilDeviceInit(argc, argv);
        if (dev < 0) {
           printf("No CUDA Capable Devices found, exiting...\n");
           shrQAFinishExit(argc, (const char **)argv, QA_WAIVED);
        }
    } else {
        cudaSetDevice( dev = cutGetMaxGflopsDeviceId() );
        cutilSafeCall( cudaChooseDevice(&dev, &deviceProp) );
    }
    cutilSafeCall( cudaGetDeviceProperties(&deviceProp, dev) );

	printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n", 
		deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

	int version = deviceProp.major * 0x10 + deviceProp.minor;

	if(version < 0x11) 
    {
        printf("There is no device supporting a minimum of CUDA compute capability 1.1 for this SDK sample\n");
        cutilDeviceReset();
		shrQAFinishExit(argc, (const char **)argv, QA_WAIVED);
    }

    cutilCheckError(cutCreateTimer(&hTimer));
    cutilCheckError(cutCreateTimer(&transferTimer));
    // Optional Command-line multiplier to increase size of array to histogram
    if (shrGetCmdLineArgumentu(argc, (const char**)argv, "sizemult", &uiSizeMult))
    {
        uiSizeMult = CLAMP(uiSizeMult, 1, 10);
        byteCount *= uiSizeMult;
    }

    shrLog("Initializing data...\n");
        shrLog("...allocating CPU memory.\n");
            h_Data         = (uchar *)malloc(byteCount);
           // h_HistogramCPU = (uint  *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
           // h_HistogramGPU = (uint  *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
           h_HistogramCPU = (uint  *)malloc(HISTOGRAM64_BIN_COUNT * sizeof(uint));
	   h_HistogramGPU = (uint  *)malloc(HISTOGRAM64_BIN_COUNT * sizeof(uint));

        shrLog("...generating input data\n");
            srand(1);
            for(uint i = 0; i < byteCount; i++) 
                h_Data[i] = rand() % 256;

        shrLog("...allocating GPU memory and copying input data\n\n");
            cutilSafeCall( cudaMalloc((void **)&d_Data, byteCount  ) );
            cutilSafeCall( cudaMalloc((void **)&d_Histogram, HISTOGRAM64_BIN_COUNT * sizeof(uint) ) );
	    cutilCheckError( cutStartTimer(transferTimer) );
            cutilSafeCall( cudaMemcpy(d_Data, h_Data, byteCount, cudaMemcpyHostToDevice) );
	   cutilCheckError( cutStopTimer(transferTimer) );
    {
        shrLog("Starting up 64-bin histogram...\n\n");
            initHistogram64(atomicAddFlag);

        shrLog("Running 64-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount, numRuns);
            for(int iter = -1; iter < numRuns; iter++){
              //iter == -1 -- warmup iteration
                if(iter == 0){
                    cutilSafeCall( cutilDeviceSynchronize() );
                    cutilCheckError( cutResetTimer(hTimer) );
                    cutilCheckError( cutStartTimer(hTimer) );
                }

                histogram64(d_Histogram, d_Data, byteCount, atomicAddFlag);
            }

            cutilSafeCall( cutilDeviceSynchronize() );
            cutilCheckError(  cutStopTimer(hTimer));
            double dAvgSecs =  (double)cutGetTimerValue(hTimer) / (double)numRuns;
        shrLog("histogram64() time (GPU, average) : %.5f ms\n", dAvgSecs);

        shrLog("\nValidating GPU results...\n");
            shrLog(" ...reading back GPU results\n");
	     cutilCheckError( cutStartTimer(transferTimer) );
             cutilSafeCall( cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM64_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost) );
	     cutilCheckError( cutStopTimer(transferTimer) );

            shrLog(" ...histogram64CPU()\n");
	    cutilCheckError( cutResetTimer(hTimer) );
            cutilCheckError( cutStartTimer(hTimer) );
            histogram64CPU(
                    h_HistogramCPU,
                    h_Data,
                    byteCount
                );
	    cutilSafeCall( cutilDeviceSynchronize() );
            cutilCheckError(  cutStopTimer(hTimer));


	dAvgSecs =  (double)cutGetTimerValue(hTimer);
        shrLog("histogram64() time (CPU) : %.5f ms\n", dAvgSecs);

	 dAvgSecs = 1.0e-3 * (double)cutGetTimerValue(transferTimer);
	shrLog("histogram64 transfer time : %.5f ms\n", dAvgSecs);
        
            shrLog(" ...comparing the results...\n");
                for(uint i = 0; i < HISTOGRAM64_BIN_COUNT; i++)
                    if(h_HistogramGPU[i] != h_HistogramCPU[i]) PassFailFlag = 0;
            shrLog(PassFailFlag ? " ...64-bin histograms match\n\n" : " ***64-bin histograms do not match!!!***\n\n" );

        shrLog("Shutting down 64-bin histogram...\n\n\n");
            closeHistogram64(atomicAddFlag);
    }

    shrLog("Shutting down...\n");
        cutilCheckError(cutDeleteTimer(hTimer));
	cutilCheckError(cutDeleteTimer(transferTimer));
        cutilSafeCall( cudaFree(d_Histogram) );
        cutilSafeCall( cudaFree(d_Data) );
        free(h_HistogramGPU);
        free(h_HistogramCPU);
        free(h_Data);

    cutilDeviceReset();
	shrLog("%s - Test Summary\n", sSDKsample);
    // pass or fail (for both 64 bit and 256 bit histograms)
    shrQAFinishExit(argc, (const char **)argv, (PassFailFlag ? QA_PASSED : QA_FAILED));
}
