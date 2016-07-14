/*
 * ECEN 5593 Summer 2016
 * Diana Southard
 *
 *
 * Vector Reduction
 * Calulates the sum of all components of a vector
 *
 */

// Includes
#include <stdio.h>
#include <cutil_inline.h>

//#define ATOMIC_ADD_FUNCTION	// define when doing part 4 of the assignment
//#define ALWAYS_EXECUTE 	// define when doing part 5 of the assignment

// Input Array Variables
float* h_In = NULL;
float* d_In = NULL;

// Output Array
float* h_Out = NULL;
float* d_Out = NULL;

// Timer Variable
static double t0=0;

// Variables to change
int GlobalSize = 50000;
int BlockSize = 32;

// Functions
void Cleanup(void);
void RandomInit(float*, int);
void PrintArray(float*, int);
float CPUReduce(float*, int);
void ParseArguments(int, char**);

/**************************************************** Device code ***********************************************************/
__global__ void VecReduce(float* input, float* output, int size)
{
	// shared memory size declared at kernel launch
	extern __shared__ float sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int globalid = blockIdx.x*blockDim.x + threadIdx.x;

	// For thread ids greater than data space
#ifdef ALWAYS_EXECUTE
	// Bypass the if statement for part 5
	sdata[tid] = input[globalid];
#else
	if (globalid < size) {
		sdata[tid] = input[globalid];
	}
	else {
		sdata[tid] = 0;  // Case of extra threads above GlobalSize
	}
#endif

	// each thread loads one element from global to shared mem
	__syncthreads();

	// do reduction in shared mem
	// using reversed loop and threadID-based indexing
	for (unsigned int s=blockDim.x / 2; s > 0; s = s >> 1) {
		if (tid < s) {
			sdata[tid] = sdata[tid] + sdata[tid+ s];
		}
		__syncthreads();
	}
#ifdef ATOMIC_ADD_FUNCTION
	// Thread 0 adds the partial sum to the total sum
	if( tid == 0 )
		atomicAdd(&output[blockIdx.x], sdata[tid]);
#else
	// write result for this block to global mem
	if (tid == 0)  {
		output[blockIdx.x] = sdata[0];
	}
#endif
}


/**************************************************** Host code ***********************************************************/
int main(int argc, char** argv)
{
	// Parse arguments, change variables as requested
	ParseArguments(argc, argv);
	printf("Vector reduction: Input Size: %d, Block Size: %d\n", GlobalSize, BlockSize);

	// determine size of data input vectors
	size_t in_size = GlobalSize * sizeof(float);
	float CPU_result = 0.0, GPU_result = 0.0;

	// Allocate input vectors h_In in host memory
	h_In = (float*)malloc(in_size);
	if (h_In == 0)
		Cleanup();

	// Initialize input vectors
	RandomInit(h_In, GlobalSize);

	// Set the kernel arguments
	int threadsPerBlock = BlockSize;
	int sharedMemSize = threadsPerBlock * sizeof(float);
	int blocksPerGrid = (GlobalSize + threadsPerBlock - 1) / threadsPerBlock;
	size_t out_size = blocksPerGrid * sizeof(float);

	// Allocate host output
	h_Out = (float*)malloc(out_size);
	if (h_Out == 0)		// problem with malloc
		Cleanup();

	// Create timers for other events
	int timerCPU = 0, timerGPU = 0, timerTransfer = 0;
	cutCreateTimer(&timerCPU);
	cutCreateTimer(&timerGPU);
	cutCreateTimer(&timerTransfer);

	// CPU computation - time this routine for base comparison
	CPU_result = CPUReduce(h_In, GlobalSize);

	// Allocate vectors in device memory
	cutilSafeCall( cudaMalloc((void**)&d_In, in_size) );
	cutilSafeCall( cudaMalloc((void**)&d_Out, out_size) );

	// Copy h_In from host memory to device memory
	cutilCheckError( cutStartTimer(timerTransfer) );
	cutilSafeCall( cudaMemcpy(d_In, h_In, in_size, cudaMemcpyHostToDevice));
	cutilCheckError( cutStopTimer(timerTransfer) );

	// Invoke kernel
	cutilCheckError( cutStartTimer(timerGPU) );	// Start CPU timer
	VecReduce<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_In, d_Out, GlobalSize);
	cutilCheckMsg("kernel launch failure");
	cutilSafeCall( cudaThreadSynchronize() ); // Have host wait for kernel
	cutilCheckError( cutStopTimer(timerGPU) );	// Start CPU timer

	// Copy results back from GPU to the h_Out
	cutilCheckError( cutStartTimer(timerTransfer) );
	cutilSafeCall( cudaMemcpy(h_Out,d_Out,out_size,cudaMemcpyDeviceToHost));
	cutilCheckError( cutStopTimer(timerTransfer) );

#ifndef ATOMIC_ADD_FUNCTION
	// Perform the CPU addition of partial results, update GPU_result
	cutilCheckError( cutStartTimer(timerCPU) );
	GPU_result = CPUReduce(h_Out, out_size);
	cutilCheckError( cutStopTimer(timerCPU) );
#endif

	// Check results to make sure they are the same
	printf("CPU results : %f\n", CPU_result);
	printf("GPU results : %f\n", GPU_result);
	printf("Timer results : \n");
	printf("\t GPU Execution Time: %d\n", cutGetTimerValue(timerGPU));
	printf("\t Memory Transfer Time: %d\n", cutGetTimerValue(timerTransfer));
	printf("\t CPU Time To Add Partial Sums: %d\n", cutGetTimerValue(timerCPU));
	printf("\t Overall Execution Time: %d\n", cutGetTimerValue(timerGPU) + cutGetTimerValue(timerTransfer) + cutGetTimerValue(timerCPU));

	// Free memory and exit
	Cleanup();
}

void Cleanup(void)
{
	// Free device memory
	if (d_In)
		cutilSafeCall( cudaFree(d_In) );
	if (d_Out)
		cutilSafeCall( cudaFree(d_Out) );

	// Free host memory
	if (h_In)
		free(h_In);
	if (h_Out)
		free(h_Out);

	cutilSafeCall( cudaThreadExit() );

	exit(0);
}

// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
	for (int i = 0; i < n; i++)
		data[i] = rand() / (float)RAND_MAX;
}

// Print through array of data
void PrintArray(float* data, int n)
{
	for (int i = 0; i < n; i++)
		printf("[%d] => %f\n",i,data[i]);
}

// CPU version of reduce
float CPUReduce(float* data, int n)
{
	float sum = 0;
	for (int i = 0; i < n; i++)
		sum = sum + data[i];

	return sum;
}

// Parse program arguments
void ParseArguments(int argc, char** argv)
{
	for (int i = 0; i < argc; ++i) {
		if (strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-size") == 0) {
			GlobalSize = atoi(argv[i+1]);
			i = i + 1;
		}
		if (strcmp(argv[i], "--blocksize") == 0 || strcmp(argv[i], "-blocksize") == 0) {
			BlockSize = atoi(argv[i+1]);
			i = i + 1;
		}
	}
}
