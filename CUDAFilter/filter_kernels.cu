
/*
 * Filter Kernel
 *
 * ECEN 5593 Summer 2016
 * Diana Southard
 *
 *
 * File containing the kernel calls for the image processing filters
 *
 */

#ifndef _FILTER_KERNELS_H_
#define _FILTER_KERNELS_H_


/*
 * Sobel Filter
 * Device code for Sobel filter (3x3 array)
 */
__global__ void DeviceSobel(unsigned char* g_DataIn, unsigned char* g_DataOut, int width, int height)
{
	__shared__ unsigned char sharedMem[BLOCK_HEIGHT * BLOCK_WIDTH];
	float s_SobelMatrix[9];

	// Set up constant Sobel Matrix
	s_SobelMatrix[0] = -1;
	s_SobelMatrix[1] = 0;
	s_SobelMatrix[2] = 1;

	s_SobelMatrix[3] = -2;
	s_SobelMatrix[4] = 0;
	s_SobelMatrix[5] = 2;

	s_SobelMatrix[6] = -1;
	s_SobelMatrix[7] = 0;
	s_SobelMatrix[8] = 1;

	// Computer the X and Y global coordinates
	int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;
	int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;

	// Get the Global index into the original image
	int index = y * (width) + x;

	// Handle the extra thread case where the image width or height
	if (x >= width || y >= height)
		return;


	// Handle the border cases of the global image
	if( x < FILTER_RADIUS || y < FILTER_RADIUS) {
		g_DataOut[index] = g_DataIn[index];
		return;
	}

	if ((x > width - FILTER_RADIUS - 1)&&(x <width)) {
		g_DataOut[index] = g_DataIn[index];
		return;
	}

	if ((y > height - FILTER_RADIUS - 1)&&(y < height)) {
		g_DataOut[index] = g_DataIn[index];
		return;
	}

	// Perform the first load of values into shared memory
	int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;
	sharedMem[sharedIndex] = g_DataIn[index];
	__syncthreads();


	// Make sure only the thread ids should write the sum of the neighbors.
	if(threadIdx.x >= (blockDim.x - FILTER_RADIUS) || threadIdx.x < FILTER_RADIUS)
		return;

	if(threadIdx.y >= (blockDim.y - FILTER_RADIUS) || threadIdx.y < FILTER_RADIUS)
		return;


	float sumX = 0, sumY = 0;
	for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
		for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
			float Pixel = (float)(sharedMem[sharedIndex + (dy * blockDim.x + dx)]);
			sumX += (float)(Pixel * s_SobelMatrix[(dy + FILTER_RADIUS) * FILTER_DIAMETER + (dx+FILTER_RADIUS)]);
			sumY += (float)(Pixel * s_SobelMatrix[(dx + FILTER_RADIUS) * FILTER_DIAMETER + (dy+FILTER_RADIUS)]);
		}
	}

	// Set to 0 or 255 based on edge threshold value
	g_DataOut[index] = (abs(sumX) + abs(sumY)) > EDGE_VALUE_THRESHOLD ? 255 : 0;
}

/*
 * Average Filter
 * Device code for Average Filter
 */
__global__ void DeviceAverage(unsigned char* g_DataIn, unsigned char* g_DataOut, int width, int height)
{
	__shared__ unsigned char sharedMem[BLOCK_HEIGHT*BLOCK_WIDTH];

	int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
	int y = blockIdx.y * TILE_HEIGHT + threadIdx.y;

	// Get the Global index into the original image
	int index = y * (width) + x;

	if (x >= width || y >= height)
		return;

	// Handle the border cases of the global image
	if( x < FILTER_RADIUS || y < FILTER_RADIUS) {
		g_DataOut[index] = g_DataIn[index];
		return;
	}

	if ((x > width - FILTER_RADIUS - 1)&&(x <width)) {
		g_DataOut[index] = g_DataIn[index];
		return;
	}

	if ((y > height - FILTER_RADIUS - 1)&&(y < height)) {
		g_DataOut[index] = g_DataIn[index];
		return;
	}

	// Perform the first load of values into shared memory
	int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;
	sharedMem[sharedIndex] = g_DataIn[index];
	__syncthreads();

	// Make sure only the thread ids should write the sum of the neighbors.
	if(threadIdx.x >= (BLOCK_WIDTH - FILTER_RADIUS) || threadIdx.x < FILTER_RADIUS)
		return;

	if(threadIdx.y >= (BLOCK_HEIGHT - FILTER_RADIUS) || threadIdx.y < FILTER_RADIUS)
		return;

	float sumX = 0;
	for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
		for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
			float Pixel = (float)(sharedMem[sharedIndex + (dy * blockDim.x + dx)]);
			sumX += Pixel;
		}
	}
	g_DataOut[index] = (unsigned char)(sumX/FILTER_AREA);
}

/*
 * Boost Filter
 * Device code for Boost filter
 */
__global__ void DeviceBoost(unsigned char* g_DataIn, unsigned char* g_DataOut, int width, int height)
{
	__shared__ unsigned char sharedMem[BLOCK_HEIGHT*BLOCK_WIDTH];

	int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;
	int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;

	// Get the Global index into the original image
	int index = y * (width) + x;

	if (x >= width || y >= height)
		return;

	// Handle the border cases of the global image
	if( x < FILTER_RADIUS || y < FILTER_RADIUS) {
		g_DataOut[index] = g_DataIn[index];
		return;
	}

	if ((x > width - FILTER_RADIUS - 1)&&(x <width)) {
		g_DataOut[index] = g_DataIn[index];
		return;
	}

	if ((y > height - FILTER_RADIUS - 1)&&(y < height)) {
		g_DataOut[index] = g_DataIn[index];
		return;
	}

	// Perform the first load of values into shared memory
	int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;
	sharedMem[sharedIndex] = g_DataIn[index];
	__syncthreads();

	if(threadIdx.x >= (BLOCK_WIDTH - FILTER_RADIUS) || threadIdx.x < FILTER_RADIUS)
		return;

	if(threadIdx.y >= (BLOCK_HEIGHT - FILTER_RADIUS) || threadIdx.y < FILTER_RADIUS)
		return;

	float sumX, centerPixel = 0;
	for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
		for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
			float Pixel = (float)(sharedMem[sharedIndex + (dy * blockDim.x + dx)]);
			sumX += Pixel;
		}
	}
	centerPixel = (float) (sharedMem[sharedIndex]);

	// Clamp between 0 and 255
	g_DataOut[index] = CLAMP_8bit(int(centerPixel + BOOST_FACTOR * (unsigned char)(centerPixel - sumX/FILTER_AREA)));
}

/*
 * SobelFilter5x5
 * Device Code for the Sobel filter using a 5x5 input array
 */

__global__ void DeviceSobel5x5(unsigned char* g_DataIn, unsigned char* g_DataOut, int width, int height, float* sobel)
{
	__shared__ unsigned char sharedMem[BLOCK_HEIGHT_5X5 * BLOCK_WIDTH_5X5];

	// Computer the X and Y global coordinates
	int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
	int y = blockIdx.y * TILE_HEIGHT + threadIdx.y;

	// Get the Global index into the original image
	int index = y * (width) + x;

	// Handle the extra thread case where the image width or height
	if (x >= width || y >= height)
		return;

	// Handle the border cases of the global image
	if( x < FILTER_RADIUS_5X5 || y < FILTER_RADIUS_5X5) {
		g_DataOut[index] = g_DataIn[index];
		return;
	}

	if ((x > width - FILTER_RADIUS_5X5 - 1)&&(x <width)) {
		g_DataOut[index] = g_DataIn[index];
		return;
	}

	if ((y > height - FILTER_RADIUS_5X5 - 1)&&(y < height)) {
		g_DataOut[index] = g_DataIn[index];
		return;
	}

	// Perform the first load of values into shared memory
	int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;
	sharedMem[sharedIndex] = g_DataIn[index];
	__syncthreads();

	// Make sure only the thread ids should write the sum of the neighbors.
	if(threadIdx.x >= (blockDim.x - FILTER_RADIUS_5X5) || threadIdx.x < FILTER_RADIUS_5X5)
		return;

	if(threadIdx.y >= (blockDim.y - FILTER_RADIUS_5X5) || threadIdx.y < FILTER_RADIUS_5X5)
		return;

	float sumX = 0, sumY = 0;
	for(int dy = -FILTER_RADIUS_5X5; dy <= FILTER_RADIUS_5X5; dy++) {
		for(int dx = -FILTER_RADIUS_5X5; dx <= FILTER_RADIUS_5X5; dx++) {
			float Pixel = (float)(sharedMem[sharedIndex + (dy * blockDim.x + dx)]);
			sumX += (float)(Pixel * sobel[(dy + FILTER_RADIUS_5X5) * FILTER_DIAMETER_5X5 + (dx+FILTER_RADIUS_5X5)]);
			sumY += (float)(Pixel * sobel[(dx + FILTER_RADIUS_5X5) * FILTER_DIAMETER_5X5 + (dy+FILTER_RADIUS_5X5)]);
		}
	}

	// Set to either 255 or 0 depending on threshhold value
	g_DataOut[index] = (0.045454545 * (abs(sumX) + abs(sumY))) > EDGE_VALUE_THRESHOLD_5X5 ? 255 : 0;
}

#endif // _FILTER_KERNELS_H_


