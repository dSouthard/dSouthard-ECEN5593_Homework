/*
 * CUDA Image Filters
 * ECEN 5593 Summer 2016
 *
 * Diana Southard
 *
 * Write the code for the Sobel, Average, and Boost Filters and use timing routines to calculate
 * the amount of time for CPU execution compared to GPU execution (including memory transfer time).
 * Then rewrite the Sobel Filter to accept a 5x5 array kernel, using the same timing routines to
 * caluclate the time savings.
 */

// Includes: system
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <errno.h>
#include <assert.h>
#include <string.h>
#include <sys/io.h>
#include <cutil_inline.h>

// Includes: helper function for reading/writing bmp files
#include "bmp.h"

// Header file containing all kernel implementations
//#include "filter_kernels.cu"

// Use enumeration to determine the type of filter being used
enum FILTERMODE {
	SOBEL, AVERAGE, BOOST, SOBEL5x5};

// Macro for clamping, used for Boost Filter
#define CLAMP_8bit(x) max(0, min(255, (x)))

// Pointers for default input/output bmp file names. These can be override at execution time
char* BMPInFile = "lena.bmp";
char* BMPOutFile = "output.bmp";

// Default FilterMode is SOBEL, keep track with FilterName for output printing
char* filterName = "sobel";
FILTERMODE filterMode = SOBEL;

// Helper Functions
void Cleanup(void);
void ParseArguments(int, char**);
void runFilterKernel(unsigned char* pImageIn, int Width, int Height);
void BitMapRead(char *file, struct bmp_header *bmp, struct dib_header *dib, unsigned char **data, unsigned char **palete);
void BitMapWrite(char *file, struct bmp_header *bmp, struct dib_header *dib, unsigned char *data, unsigned char *palete);
void PrintUsageStatement();

// CPU Image Filter Functions
void HostSobel(unsigned char* imageIn, unsigned char* imageOut, int width, int height);
void HostAverage(unsigned char* imageIn, unsigned char* imageOut, int width, int height);
void HostBoost(unsigned char* imageIn, unsigned char* imageOut, int width, int height);
void HostSobel5x5(unsigned char* imageIn, unsigned char* imageOut, int width, int height);

/* Device Memory */
unsigned char * d_In;	// Input to device
unsigned char * d_Out;	// Output from device
float * d_SobelMask;	// Input Sobel mask, used for 5x5 matrix
float * SobelMaskMatrix;	// Sobel Mask Matrix

// Setup for kernel size, both 3x3 and 5x5
const int TILE_WIDTH = 6;
const int TILE_HEIGHT = 6;

const int FILTER_RADIUS = 1;		// Sobel 3x3 = 1, 5x5 = 3
const int FILTER_RADIUS_5X5 = 3;
const int FILTER_DIAMETER = 3;		// Sobel 3x3 = 3, 5x5 = 5
const int FILTER_DIAMETER_5X5 = 5;
const int FILTER_AREA = FILTER_DIAMETER * FILTER_DIAMETER;
const int FILTER_AREA_5X5 = FILTER_DIAMETER_5X5 * FILTER_DIAMETER_5X5;

const int BLOCK_WIDTH = TILE_WIDTH + 2*FILTER_RADIUS;
const int BLOCK_HEIGHT = TILE_HEIGHT + 2*FILTER_RADIUS;

const int BLOCK_WIDTH_5X5 = TILE_WIDTH + 2*FILTER_RADIUS_5X5;
const int BLOCK_HEIGHT_5X5 = TILE_HEIGHT + 2*FILTER_RADIUS_5X5;

const int EDGE_VALUE_THRESHOLD = 70;	// Sobel 3x3 = 70, 5x5 = 800
const int EDGE_VALUE_THRESHOLD_5X5 = 800;
const int BOOST_FACTOR = 10;

//Timer variables
unsigned int timer_GPU = 0;
unsigned int timer_CPU = 0;

// Include kernel definitions
#include "filter_kernels.cu"

//#define DEBUG

// Host code
int main(int argc, char** argv)
{
	ParseArguments(argc, argv);
	struct bmp_header bmp;
	struct dib_header dib;

	unsigned char *palete = NULL;
	unsigned char *data = NULL, *out = NULL;

	// Create timers to compare execution times
	cutilCheckError(cutCreateTimer(&timer_CPU));
	cutilCheckError(cutCreateTimer(&timer_GPU));

	// Read in BMPs
	printf("Running %s filter\n", filterName);
	BitMapRead(BMPInFile, &bmp, &dib, &data, &palete);
	out = (unsigned char *)malloc(dib.image_size);

	printf("Image details: %d by %d = %d , imagesize = %d\n", dib.width, dib.height, dib.width * dib.height,dib.image_size);
#ifdef DEBUG
	printf("Computing the CPU output\n");
#endif

	/**************** Run the CPU Image Filters **/	
	cutilCheckError(cutStartTimer(timer_CPU));		// Start CPU Timer
	switch(filterMode){
	case AVERAGE:
#ifdef DEBUG
		printf("Running the Average filter on the CPU\n");
#endif
		HostAverage(data, out, dib.width, dib.height);
		break;
	case BOOST:
#ifdef DEBUG
		printf("Running the Boost filter on the CPU\n");
#endif
		HostBoost(data, out, dib.width, dib.height);
		break;
	case SOBEL5x5:
#ifdef DEBUG
		printf("Running the Sobel filter with a 5x5 kernel on the CPU\n");
#endif
		HostSobel5x5(data, out, dib.width, dib.height);
		break;
	default:
		// SOBEL is the default
#ifdef DEBUG
		printf("Running the Sobel filter with a 3x3 kernel on the CPU\n");
#endif
		HostSobel(data, out, dib.width, dib.height);
		break;
	}
	cutilCheckError(cutStopTimer(timer_CPU));		// Stop CPU Timer

	// Write the CPU Output Image
	BitMapWrite("CPU_output.bmp", &bmp, &dib, out, palete);
	
	// Print out new details
#ifdef DEBUG
	printf("Done with CPU output\n");
#endif
	/**************** Run the GPU Image Filter */
#ifdef DEBUG
	printf("Allocating %d bytes for image \n", dib.image_size);
#endif
	cutilSafeCall( cudaMalloc( (void **)&d_In, dib.image_size*sizeof(unsigned char)) );
	cutilSafeCall( cudaMalloc( (void **)&d_Out, dib.image_size*sizeof(unsigned char)) );
	if (filterMode == SOBEL5x5){
		// Allocate space for input Sobel mask array to device
		cutilSafeCall( cudaMalloc( (void **)&d_SobelMask, FILTER_AREA_5X5*sizeof(float)) );
	}

	// Start the GPU Timer to capture memory transfer and all computation execution time
	cutilCheckError(cutStartTimer(timer_GPU));
	cutilSafeCall( cudaMemcpy(d_In, data, dib.image_size*sizeof(unsigned char), cudaMemcpyHostToDevice) );
	if (filterMode == SOBEL5x5) {
		// Copy Sobel mask to device
		cudaMemcpy(d_SobelMask, SobelMaskMatrix, FILTER_AREA_5X5*sizeof(float), cudaMemcpyHostToDevice);
	}

	// Call the correct kernel
	runFilterKernel(data, dib.width, dib.height);

	// Copy image back to host
	cutilSafeCall( cudaMemcpy(out, d_Out, dib.image_size*sizeof(unsigned char), cudaMemcpyDeviceToHost) );
	cutilCheckError(cutStopTimer(timer_GPU));		// Stop the GPU Timer

	// Write GPU Output image
	BitMapWrite(BMPOutFile, &bmp, &dib, out, palete);
	// Print out new image details
	//printf("GPU filtered image details: %d by %d = %d , imagesize = %d\n", dib.width, dib.height, dib.width * dib.height,dib.image_size);

	// Print out results of timing routines
	printf("CPU Execution time : %f (ms) \n",cutGetTimerValue(timer_CPU));
	printf("GPU Execution time : %f (ms) \n\n",cutGetTimerValue(timer_GPU));

	Cleanup();
}

// Free up memory and timers
void Cleanup(void)
{
	// Free device memory
	if (d_In)
		cutilSafeCall( cudaFree(d_In) );
	if (d_Out)
		cutilSafeCall( cudaFree(d_Out) );

	// Free SobelMask memory
	if (d_SobelMask)
		free(d_SobelMask);
	if (SobelMaskMatrix)
		delete[] SobelMaskMatrix;

	//Destroy timer values
	cutilCheckError(cutDeleteTimer(timer_GPU));
	cutilCheckError(cutDeleteTimer(timer_CPU));

	// Exit the program
	cutilSafeCall( cudaThreadExit() );
	exit(0);
}

// Set up
void runFilterKernel(unsigned char* pImageIn, int Width, int Height){
	// Design grid disection around tile size
	int gridWidth  = (Width + TILE_WIDTH - 1) / TILE_WIDTH;
	int gridHeight = (Height + TILE_HEIGHT - 1) / TILE_HEIGHT;
	dim3 dimGrid(gridWidth, gridHeight);

	

	// Invoke larger blocks to take care of surrounding shared memory
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
	if (filterMode == SOBEL5x5){
		dimBlock.x = BLOCK_WIDTH_5X5;
		dimBlock.y = BLOCK_HEIGHT_5X5;
	}

	switch(filterMode) {
	case SOBEL5x5:
#ifdef DEBUG		
		printf("Running the Sobel Filter with a 5x5 kernel on the GPU \n");
#endif
		DeviceSobel5x5<<< dimGrid, dimBlock >>>(d_In, d_Out, Width, Height, d_SobelMask);
		cutilCheckMsg("kernel launch failure");
		break;
	case AVERAGE:
#ifdef DEBUG
		printf("Running the Average Filter on the GPU \n");
#endif
		DeviceAverage<<< dimGrid, dimBlock >>>(d_In, d_Out, Width, Height);
		cutilCheckMsg("kernel launch failure");
		break;
	case BOOST:
#ifdef DEBUG
		printf("Running the Boost Filter on the GPU \n");
#endif
		DeviceBoost<<< dimGrid, dimBlock >>>(d_In, d_Out, Width, Height);
		cutilCheckMsg("kernel launch failure");
		break;
	default:
#ifdef DEBUG
		printf("Running the Sobel Filter with a 3x3 kernel on the GPU\n");
#endif
		DeviceSobel<<< dimGrid, dimBlock >>>(d_In, d_Out, Width, Height);
		cutilCheckMsg("kernel launch failure");
		break;
	}
	cutilSafeCall( cudaThreadSynchronize() );
}

// Parse program arguments
void ParseArguments(int argc, char** argv){
#ifdef DEBUG
	printf("Number of incoming arguments: %d\n", argc);
#endif

	for (int i = 1; i < argc; ++i) {
#ifdef DEBUG
	printf("i = %d, argv[i] = %s, argv[i+1] = %c\n", i, argv[i], argv[i+1]);
#endif
		// Change the name of the input file
		if (strcmp(argv[i], "--file") == 0 || strcmp(argv[i], "-file") == 0) {
			BMPInFile = argv[i+1];
#ifdef DEBUG
			printf("input is %s, \n", BMPInFile);
#endif

		}
		// Change the name of the output file
		else if (strcmp(argv[i], "--out") == 0 || strcmp(argv[i], "-out") == 0) {
			BMPOutFile = argv[i+1];
#ifdef DEBUG
			printf("output is %s, \n", BMPOutFile);
#endif
		}
		// Change type of filtering being used
		else if (strcmp(argv[i], "--filter") == 0 || strcmp(argv[i], "-filter") == 0) {
			filterName = argv[i+1];
#ifdef DEBUG
			printf("filter is %s, \n", filterName);
#endif

			if ((strcmp(filterName, "sobel") == 0)|| strcmp(argv[i], "Sobel") == 0){
				filterMode = SOBEL;
				// Setup SobelMaskMatrix
				SobelMaskMatrix = new  float[9];
				SobelMaskMatrix[0] = -1; SobelMaskMatrix[1] = 0; SobelMaskMatrix[2] =  1; 
				SobelMaskMatrix[3] = -2; SobelMaskMatrix[4] = 0; SobelMaskMatrix[5] = 2;
				SobelMaskMatrix[6] = -1; SobelMaskMatrix[7] = 0; SobelMaskMatrix[8] = 1;
#ifdef DEBUG
			if (filterMode == SOBEL) printf("filter mode = sobel, mask set up\n");
#endif
			}
			else if ((strcmp(filterName, "average") == 0)|| strcmp(argv[i], "Average") == 0){
				filterMode = AVERAGE;
#ifdef DEBUG
				if (filterMode == AVERAGE) printf("filtermode is average\n");
#endif

			}
			else if ((strcmp(filterName, "boost") == 0)|| strcmp(argv[i], "Boost") == 0){
				filterMode = BOOST;
#ifdef DEBUG
				if (filterMode == BOOST) printf("filtermode is boost\n");
#endif
			}
			else if (strcmp(filterName, "5x5") == 0) {
				filterMode = SOBEL5x5;
				// Setup SobelMaskMatrix
				SobelMaskMatrix = new  float[25];
                                SobelMaskMatrix[0] = 1;  SobelMaskMatrix[1] = 2;   SobelMaskMatrix[2] = 0;  SobelMaskMatrix[3] = -2;   SobelMaskMatrix[4] = -1;
                                SobelMaskMatrix[5] = 4;  SobelMaskMatrix[6] = 8;   SobelMaskMatrix[7] = 0;  SobelMaskMatrix[8] = -8;   SobelMaskMatrix[9] = -4;
                                SobelMaskMatrix[10] = 6; SobelMaskMatrix[11] = 12; SobelMaskMatrix[12] = 0; SobelMaskMatrix[13] = -12; SobelMaskMatrix[14] = -6;
				SobelMaskMatrix[15] = 4; SobelMaskMatrix[16] = 8;  SobelMaskMatrix[17] = 0; SobelMaskMatrix[18] = -8;  SobelMaskMatrix[19] = -4;
				SobelMaskMatrix[20] = 1; SobelMaskMatrix[21] = 2;  SobelMaskMatrix[22] = 0; SobelMaskMatrix[23] = -2;  SobelMaskMatrix[24] = -1;
			//	SobelMaskMatrix = {1, 2, 0, -2, -1, 4, 8, 0, -8, -4, 6, 12, 0, -12, -6, 4, 8, 0, -8, -4, 1, 2, 0, -2, -1};
#ifdef DEBUG
				if (filterMode == SOBEL5x5) printf("filter mode = sobel5x5, mask set up\n");
#endif

			}
			else if (strcmp(filterName, "5x5a") == 0) {
				filterMode = SOBEL5x5;
				// Setup second SobelMaskMatrix
				SobelMaskMatrix = new  float[25];
                                SobelMaskMatrix[0] = -1; SobelMaskMatrix[1] = -4; SobelMaskMatrix[2] = -6;  SobelMaskMatrix[3] = -4; SobelMaskMatrix[4] = -1;
                                SobelMaskMatrix[5] = -2; SobelMaskMatrix[6] = -8; SobelMaskMatrix[7] = -12; SobelMaskMatrix[8] = -8; SobelMaskMatrix[9] = -2;
                                SobelMaskMatrix[10] = 0; SobelMaskMatrix[11] = 0; SobelMaskMatrix[12] = 0;  SobelMaskMatrix[13] = 0; SobelMaskMatrix[14] = 0;
                                SobelMaskMatrix[15] = 2; SobelMaskMatrix[16] = 8; SobelMaskMatrix[17] = 12; SobelMaskMatrix[18] = 8; SobelMaskMatrix[19] = 2;
                                SobelMaskMatrix[20] = 1; SobelMaskMatrix[21] = 4; SobelMaskMatrix[22] = 6;  SobelMaskMatrix[23] = 4; SobelMaskMatrix[24] = 1;
			//	SobelMaskMatrix = {-1, -4, -6, -4, -1, -2, -8, -12, -8, -2, 0, 0, 0, 0, 0, 2, 8, 12, 8, 2, 1, 4, 6, 4, 1};
#ifdef DEBUG
				if (filterMode == SOBEL5x5) printf("filter mode = sobel, second mask set up\n");
#endif

			}
			else{
				// Print usage statement and exit
				PrintUsageStatement();
				Cleanup();
			}
		}
		// Look at next incoming argument, skipping argument after flag
		i++;
	}
	// Check that SobelMaskMatrix was correctly set up, default in case there was no incoming argument
	if (filterMode == SOBEL && SobelMaskMatrix == NULL){
        // Setup SobelMaskMatrix
        SobelMaskMatrix = new  float[9];
        SobelMaskMatrix[0] = -1; SobelMaskMatrix[1] = 0; SobelMaskMatrix[2] =  1;
        SobelMaskMatrix[3] = -2; SobelMaskMatrix[4] = 0; SobelMaskMatrix[5] = 2;
        SobelMaskMatrix[6] = -1; SobelMaskMatrix[7] = 0; SobelMaskMatrix[8] = 1;
#ifdef DEBUG
	if (filterMode == SOBEL) printf("filter mode = sobel, mask set up, not originally set up\n");
#endif

        }
}

/***************************** BMP Helper Functions ****************************/
// Read in bmp file and convert to useable data
void BitMapRead(char *file, struct bmp_header *bmp, struct dib_header *dib, unsigned char **data, unsigned char **palete)
{
	size_t palete_size;
	int fd;

	if((fd = open(file, O_RDONLY )) < 0)
		FATAL("Open Source");

	if(read(fd, bmp, BMP_SIZE) != BMP_SIZE)
		FATAL("Read BMP Header");

	if(read(fd, dib, DIB_SIZE) != DIB_SIZE)
		FATAL("Read DIB Header");

	assert(dib->bpp == 8);

	palete_size = bmp->offset - BMP_SIZE - DIB_SIZE;
	if(palete_size > 0) {
		*palete = (unsigned char *)malloc(palete_size);
		int go = read(fd, *palete, palete_size);
		if (go != palete_size) {
			FATAL("Read Palete");
		}
	}

	*data = (unsigned char *)malloc(dib->image_size);
	if(read(fd, *data, dib->image_size) != dib->image_size)
		FATAL("Read Image");

	close(fd);
}

// Write out data to viewable bmp image
void BitMapWrite(char *file, struct bmp_header *bmp, struct dib_header *dib, unsigned char *data, unsigned char *palete)
{
	size_t palete_size;
	int fd;

	palete_size = bmp->offset - BMP_SIZE - DIB_SIZE;

	if((fd = open(file, O_WRONLY | O_CREAT | O_TRUNC,
			S_IRUSR | S_IWUSR |S_IRGRP)) < 0)
		FATAL("Open Destination");

	if(write(fd, bmp, BMP_SIZE) != BMP_SIZE)
		FATAL("Write BMP Header");

	if(write(fd, dib, DIB_SIZE) != DIB_SIZE)
		FATAL("Write BMP Header");

	if(palete_size != 0) {
		if(write(fd, palete, palete_size) != palete_size)
			FATAL("Write Palete");
	}
	if(write(fd, data, dib->image_size) != dib->image_size)
		FATAL("Write Image");
	close(fd);
}

/***************************** CPU Image Filter Functions *******************************/
// Sobel Filter: calculates the gradient of the image intensity at each point, used for edge-detection
void HostSobel(unsigned char* imageIn, unsigned char* imageOut, int width, int height) {
	int row, column;
	// Initialize all output pixels to zero
	for(row = 0; row < height; row++) {
		for(column = 0; column < width; column++) {
			imageOut[row*width + column] = 0;
		}
	}

	row = 1;
	column = 1;

	// Go through all inner pixel positions
	for(row = 1; row < height - 1; row++) {
		for(column = 1; column < width - 1; column++) {
			// sum up the SobelMatrix values to calculate both the direction x and direction y
			float sumX = 0, sumY=0;
			for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
				for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
					float Pixel = (float)(imageIn[row*width + column +  (dy * width + dx)]);
					sumX += Pixel * SobelMaskMatrix[(dy + FILTER_RADIUS) * FILTER_DIAMETER + (dx+FILTER_RADIUS)];		
			sumY += Pixel * SobelMaskMatrix[(dx + FILTER_RADIUS) * FILTER_DIAMETER + (dy+FILTER_RADIUS)];
				}
			}
			// Set pixel to a 1 or a 0
			imageOut[row*width + column] = (abs(sumX) + abs(sumY)) > EDGE_VALUE_THRESHOLD ? 255 : 0;
		}
	}
}

// Sobel Filter with 5x5 Kernel
void HostSobel5x5(unsigned char* imageIn, unsigned char* imageOut, int width, int height) {
	int row, column;

	// Initialize all output pixels to zero
	for (row = 0; row < height; row++)
		for (column = 0; column < width; column++)
			imageOut[row*width + column] = 0;
	
	// Go through all inner pixel positions
	for (row = 1; row < height - 1; row++){
		for (column = 1; column < width - 1; column++){
			// sum up all SobelMatrix values
			float sumX = 0, sumY = 0;
			for (int dy = -FILTER_RADIUS_5X5; dy < FILTER_RADIUS_5X5; dy++) {
				for (int dx = -FILTER_RADIUS_5X5; dx < FILTER_RADIUS_5X5; dx++){
					float Pixel = (float)imageIn[row*width+column + (dy*width + dx)];
					sumX += Pixel * SobelMaskMatrix[(dy + FILTER_RADIUS_5X5)*FILTER_DIAMETER_5X5 + (dx + FILTER_RADIUS_5X5)];
					sumY += Pixel * SobelMaskMatrix[(dx + FILTER_RADIUS_5X5)*FILTER_DIAMETER_5X5 + (dy + FILTER_RADIUS_5X5)];
				}
			}
			// Set pixel to 1 or 0
			imageOut[row*width+column] =( (abs(sumX)+abs(sumY)) > EDGE_VALUE_THRESHOLD_5X5) ? 255:0;
		}
	}
}

// Average Filter: each pixel generates the sum of eight neighbors and the center pixel, and divides by 9
void HostAverage(unsigned char* imageIn, unsigned char* imageOut, int width, int height) {
	int row, column;

	// Initialize all output pixels to zero
	for(row = 0; row < height; row++) {
		for(column = 0; column < width; column++) {
			imageOut[row*width + column] = 0;
		}
	}

	// Go through all inner pixel positions
	for(row = 1; row < height - 1; row++) {
		for(column = 1; column < width - 1; column++) {
			// sum up the 9 values to calculate both the direction x and direction y
			float sumX = 0;
			for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
				for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
					float Pixel = (float)(imageIn[row*width + column +  (dy * width + dx)]);
					sumX+=Pixel;
				}
			}
			imageOut[row*width + column] = (unsigned char)(sumX/FILTER_AREA);
		}
	}
}

// Boost Filter: sums up 9 values in filter then subtracts the difference between the center pixel and average. The difference
// is then multiplied by a BOOST_FACTOR and added to the original value, guarenteeed to be between 0 and 255
void HostBoost(unsigned char* imageIn, unsigned char* imageOut, int width, int height) {
	int row, column;

	// Initialize all output pixels to zero
	for(row = 0; row < height; row++) {
		for(column = 0; column < width; column++) {
			imageOut[row*width + column] = 0;
		}
	}

	// Go through all inner pixel positions
	for(row = 1; row < height - 1; row++) {
		for(column = 1; column < width - 1; column++) {
			// sum up the 9 values to calculate both the direction x and direction y
			float sumX,centerPixel = 0;
			for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
				for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
					float Pixel = (float)(imageIn[row*width + column +  (dy * width + dx)]);
					sumX+=Pixel;
				}
			}

			centerPixel = imageIn[row*width + column];
			// Clamp between 0 and 255
			imageOut[row*width + column]  = CLAMP_8bit(int(centerPixel+ (BOOST_FACTOR * (unsigned char)(centerPixel - (sumX / FILTER_AREA)))));
		}
	}
}

// Print brief statement about program usage and optional flags
void PrintUsageStatement() {
	printf("CUDA Filter Assignment, ECEN 5593 - Summer 2016\n"
			" by Diana Southard\n"
			"USAGE: ./filter --file --out --filter \n"
			"Default input file: lena.bmp  [MUST BE IN SAME FILE AS EXECUTABLE]\n"
			"Default output file: output.bmp (generated by GPU) and CPU_output.bmp (generated by CPU)\n"
			"\n \t FLAGS:\n"
			"\t --file, -file: change the desired input image.\n"
			"\t --out, -out: change the desired output image name.\n"
			"\t --filter, -filter: change the desired image filter being used [MUST SELECT FROM THE FOLLOWING]:\n"
			"\t \t sobel: Use the sobel filter with a 3x3 kernel size [DEFAULT]\n"
			"\t \t \t Sobel Mask: {-1, 0, 1, -2, 0, 2, -1, 0, 1}\n"
			"\t \t average: Use the average filter with a 3x3 kernel size\n"
			"\t \t boost: Use the boost filter with a 3x3 kernel size\n"
			"\t \t sobel5x5: Use the sobel filter with a 5x5 kernel size\n"
			"\t \t \t Sobel Mask: {1, 2, 0, -2, -1, 4, 8, 0, -8, -4, 6, 12, 0, -12, -6, 4, 8, 0, -8, -4, 1, 2, 0, -2, -1}\n"
			"\t \t sobel5x5a: Use the sobel filter with a 5x5 kernel size\n"
			"\t \t \t Sobel Mask: {-1, -4, -6, -4, -1, -2, -8, -12, -8, -2, 0, 0, 0, 0, 0, 2, 8, 12, 8, 2, 1, 4, 6, 4, 1}\n"
			"\n END USAGE\n");
}


