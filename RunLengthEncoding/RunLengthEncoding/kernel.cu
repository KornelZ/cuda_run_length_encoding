
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>

//#define INPUT_SIZE 73728
#define INPUT_SIZE 39942400
#define GRID_X 395
#define GRID_Y 395
#define GRID_Z 1
#define BLOCK_X 256
#define BLOCK_Y 1
#define BLOCK_Z 1

cudaError_t runLengthEncoding(char **outText, int **outAmount, int **temp, char **in, unsigned int size, int *outSize);
//example input: a, b, b, c, c, c, d, e, e
//flags:		 1, 1, 0, 1, 0, 0, 1, 1, 0
//prefix sum:    1, 2, 2, 3, 3, 3, 4, 5, 5
__global__ void setFlags(int *tmpArr, const char *text, int size)
{
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int i = blockId * blockDim.x + threadIdx.x;
    if(i == 0 || text[i] != text[i - 1]) {
		tmpArr[i] = 1;
	}
	//prefix sum(naive)
	//int val = 0;
	//for(int j = 0, offset = 1; j <= logSize + 1; j++, offset *= 2)
	//{
	//	__syncthreads();
	//	if(i + offset < size)
	//	{
	//		val = tmpArr[i] + tmpArr[i + offset];
	//	}
	//	tmpArr[i + offset] = val;
	//}
}
//i:             0, 1, 2, 3, 4, 5, 6, 7, 8
//example input: a, b, b, c, c, c, d, e, e
//prefix sum:    1, 2, 2, 3, 3, 3, 4, 5, 5
//index:         0, 1, -, 2, -, -, 3, 4, -
//outText:       a, b, c, d, e
//Amount I:      0, 1, 3, 6, 7 --> 1 - 0, 3 - 1, 6 - 3, 7 - 6, (size) - 7 <- make it use thrust::inclusive_scan
//Amount II:     1, 2, 3, 1, 2 --> 1,     2,   , 3    , 1    , 2
__global__ void encode(int *outAmount, char *outText, int *prefixSum, const char *text, int outSize, int inSize)
{
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int i = blockId * blockDim.x + threadIdx.x;
	if(i == 0 || prefixSum[i] > prefixSum[i - 1]) {
		int index = prefixSum[i] - 1;
		outText[index] = text[i];
		outAmount[index] = i;
		//__syncthreads();
		//int amount = 0;
		//if(prefixSum[i] < outSize)
		//{
		//	amount = outAmount[prefixSum[i]] - outAmount[index];
		//}
		//else
		//{
		//	amount = inSize - outAmount[index]; 
		//}
		//__syncthreads();
		//outAmount[index] = amount;
	}
}

void printArray(char *msg, int *arr, int size)
{
	printf("%s: ", msg);
	for(int i = 0; i < size; i++) {
		printf("%d, ", arr[i]);
	}
	putchar('\n');
}

void printCharArray(char *msg, char *arr, int size)
{
	printf("%s: ", msg);
	for(int i = 0; i < size; i++) {
		printf("%c, ", arr[i]);
	}
	putchar('\n');
}

int initializeArray(int **arr, int size, bool fillRandom)
{
	*arr = (int*)malloc(size * sizeof(int));

	if(*arr == NULL) {
		return 1;
	}

	if(fillRandom) {
		srand(NULL);
		for(int i = 0; i < size; i++) {
			(*arr)[i] = rand() % 5 + 63;
		}
	} else {
		for(int i = 0; i < size; i++) {
			(*arr)[i] = 0;
		}
		return 0;
	}
	return 0;
}

int initializeCharArray(char **arr, int size, bool fillRandom)
{
	*arr = (char*)malloc(size * sizeof(char));

	if(*arr == NULL) {
		return 1;
	}

	if(fillRandom) {
		srand(NULL);
		for(int i = 0; i < size; i++) {
			(*arr)[i] = (char)(rand() % 5 + 'A');
		}
		(*arr)[size - 1] = (char)(((*arr)[size - 2] + 32) % 256);

	} else {
		for(int i = 0; i < size; i++) {
			(*arr)[i] = 0;
		}

		return 0;
	}
	return 0;
}

void checkOutput(char *input, char *outText, int *outAmount, int outputSize)
{
	bool outputOk = true;
	long errorCount = 0;
	for(int outI = 0, inI = 0; outI < outputSize && inI < INPUT_SIZE; outI++)
	{
		if(input[inI] == outText[outI]) {
			int count = inI + outAmount[outI + 1];
			while(inI < count) {
				//printf("Input: %c, Output: %c, Amount: %d\n", input[j], outText[i], outAmount[i + 1]);
				if(input[inI] != outText[outI]) {
					printf("Error at %d in outAmount\n", outI + 1);
					errorCount++;
					outputOk = false;
				}
				inI++;
			}
		} else {
			printf("Error at %d in outText\n", outI);
			errorCount++;
			outputOk = false;
		}
	}
	if(outputOk) {
		printf("Output checked: no errors\n");
	} else {
		printf("Output checked: total %l errors\n", errorCount);
	}
}

int handleError(cudaError_t error, char *errorMsg)
{
	if(error != cudaSuccess) {
		fprintf(stderr, "%s; error: %s\n", errorMsg, cudaGetErrorString(error));
		return 1;
	}
	return 0;
}

cudaError_t startTimer(cudaEvent_t *start, cudaEvent_t *stop)
{
	cudaError_t error;
	if(handleError(error = cudaEventCreate(start), "cudaEventCreate start failed")) { return error; }
	if(handleError(error = cudaEventCreate(stop), "cudaEventCreate stop failed")) { return error; }
	if(handleError(error = cudaEventRecord(*start), "cudaEventRecord start failed")) { return error; }

	return error;
}

cudaError_t stopTimer(cudaEvent_t *start, cudaEvent_t *stop, char *eventName, float *totalTime)
{
	cudaError_t error;
	float milliseconds = 0;
	if(handleError(error = cudaEventRecord(*stop), "cudaEventRecord stop failed")) { return error; }
	if(handleError(error = cudaEventSynchronize(*stop), "cudaEventSynchronize failed")) { return error; }
	if(handleError(error = cudaEventElapsedTime(&milliseconds, *start, *stop), "cudaEventElapsedTime failed")) { return error; }
	printf("%f elapsed time : %s\n", milliseconds, eventName);

	*totalTime += milliseconds;
	return error;
}

void showGpu()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("Major revision number:         %d\n",  prop.major);
    printf("Minor revision number:         %d\n",  prop.minor);
    printf("Name:                          %s\n",  prop.name);
    printf("Total global memory:           %u\n",  prop.totalGlobalMem);
    printf("Total shared memory per block: %u\n",  prop.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  prop.regsPerBlock);
    printf("Warp size:                     %d\n",  prop.warpSize);
    printf("Maximum memory pitch:          %u\n",  prop.memPitch);
    printf("Maximum threads per block:     %d\n",  prop.maxThreadsPerBlock);
    for (int i = 0; i < 3; i++) {
		printf("Maximum dimension %d of block:  %d\n", i, prop.maxThreadsDim[i]);
	}
    for (int i = 0; i < 3; i++) {
		printf("Maximum dimension %d of grid:   %d\n", i, prop.maxGridSize[i]);
	}
    printf("Clock rate:                    %d\n",  prop.clockRate);
    printf("Total constant memory:         %u\n",  prop.totalConstMem);
    printf("Texture alignment:             %u\n",  prop.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (prop.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  prop.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (prop.kernelExecTimeoutEnabled ? "Yes" : "No"));
}
int main()
{
	int arraySize = INPUT_SIZE;
    char *input = 0;
	int *prefix = 0;
	int *outAmount = 0;
	char *outText = 0; 
	showGpu();
	int outSize = 0;
	int *pOutSize = &outSize;

	if(initializeCharArray(&input, arraySize, true)) { printf("Error malloc input\n"); return 1; }
	if(initializeArray(&prefix, arraySize, false)) { printf("Error malloc input\n"); return 1; }

	if(runLengthEncoding(&outText, &outAmount, &prefix, &input, arraySize, pOutSize) != cudaSuccess) {
		fprintf(stderr, "runLengthEncoding failed");
		return 1;
	}

	/*printCharArray("Input: ", input, arraySize);
	printArray("Prefix sum: ", prefix, arraySize);
	printCharArray("Values in input: ", outText, outSize);
	printArray("Amount of consecutive values: ", outAmount, outSize);*/
	checkOutput(input, outText, outAmount, outSize);
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();

	if(handleError(cudaDeviceReset(), "cudaDeviceReset failed!")) {
		return 1;
	}

	free(input);
	free(prefix);
	free(outText);
	free(outAmount);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t runLengthEncoding(char **outText, int **outAmount, int **temp, char **in, unsigned int size, int *outSize)
{
    char *dev_in = 0;
	int *dev_temp = 0;
	int *dev_amount = 0;
	char *dev_text = 0;
	dim3 gridSize(GRID_X, GRID_Y, GRID_Z);
	dim3 blockSize(BLOCK_X, BLOCK_Y, BLOCK_Z);
	float milliseconds = 0;

    cudaError_t cudaStatus;
	cudaEvent_t start, stop;

	if(startTimer(&start, &stop) != cudaSuccess) { goto Error; }
    // Choose which GPU to run on, change this on a multi-GPU system.
	if (handleError(cudaStatus = cudaSetDevice(0), "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?")) {
        goto Error;
    }
    // Allocate GPU buffers for three vectors (two input, one output)    .
    if(handleError(cudaStatus = cudaMalloc((void**)&dev_in, size * sizeof(int)), "cudaMalloc dev_in failed")) {
		goto Error;
	}

	if(handleError(cudaStatus = cudaMalloc((void**)&dev_temp, size * sizeof(int)), "cudaMalloc dev_temp failed")) {
		goto Error;
	}

    // Copy input vectors from host memory to GPU buffers.
    if(handleError(cudaStatus = cudaMemcpy(dev_in, *in, size * sizeof(char), cudaMemcpyHostToDevice), "cudaMemcpy dev_in failed")) {
		goto Error;
	}

    if(handleError(cudaStatus = cudaMemcpy(dev_temp, *temp, size * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy dev_temp failed")) {
		goto Error;
	}

	if(stopTimer(&start, &stop, "memcpy", &milliseconds) != cudaSuccess) { goto Error; }
	if(startTimer(&start, &stop) != cudaSuccess) { goto Error; }

    // Launch a kernel on the GPU with one thread for each element.
	setFlags<<<gridSize, blockSize>>>(dev_temp, dev_in, size);

    // Check for any errors launching the kernel
    if(handleError(cudaStatus = cudaGetLastError(), "setFlags failed")) {
		goto Error;
	}
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    if(handleError(cudaStatus = cudaDeviceSynchronize(), "cudaSynchronize setFlags failed")) {
		goto Error;
	}

	if(stopTimer(&start, &stop, "setFlags", &milliseconds) != cudaSuccess) { goto Error; }
	if(startTimer(&start, &stop) != cudaSuccess) { goto Error; }
	
	thrust::device_ptr<int> temp_ptr = thrust::device_pointer_cast<int>(dev_temp);
	thrust::inclusive_scan(thrust::device, temp_ptr, temp_ptr + size, temp_ptr);
	
	if(stopTimer(&start, &stop, "scan", &milliseconds) != cudaSuccess) { goto Error; }
	if(startTimer(&start, &stop) != cudaSuccess) { goto Error; }
     //Copy size of output from GPU to memory.
    if(handleError(cudaStatus = cudaMemcpy(*temp + size - 1, dev_temp + size - 1, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy size failed")) {
		goto Error;
	}

	if(stopTimer(&start, &stop, "memcpy", &milliseconds) != cudaSuccess) { goto Error; }
	if(startTimer(&start, &stop) != cudaSuccess) { goto Error; }
	//find size of output array
	int outputSize = (*temp)[size - 1];
	(*temp)[size - 1] = INPUT_SIZE;
	*outSize = outputSize;
	if(initializeCharArray(outText, outputSize, false)) { printf("Error malloc outText %d outputSize\n", outputSize); goto Error; }
	if(initializeArray(outAmount, outputSize, false)) { printf("Error malloc outAmount %d outputSize\n", outputSize); goto Error; }

	if(handleError(cudaStatus = cudaMalloc((void**)&dev_amount, outputSize * sizeof(int)), "cudaMalloc dev_amount failed")) {
		goto Error;
	}
	if(handleError(cudaStatus = cudaMalloc((void**)&dev_text, outputSize * sizeof(char)), "cudaMalloc dev_text failed")) {
		goto Error;
	}

	encode<<<gridSize, blockSize>>>(dev_amount, dev_text, dev_temp, dev_in, outputSize, size);

	if(handleError(cudaStatus = cudaGetLastError(), "encode failed")) {
		goto Error;
	}	
	if(handleError(cudaStatus = cudaDeviceSynchronize(), "cudaSynchronize encode failed")) {
		goto Error;
	}

	if(stopTimer(&start, &stop, "encode", &milliseconds) != cudaSuccess) { goto Error; }
	if(startTimer(&start, &stop) != cudaSuccess) { goto Error; }

	thrust::device_ptr<int> amount_ptr = thrust::device_pointer_cast<int>(dev_amount);
	thrust::adjacent_difference(thrust::device, amount_ptr, amount_ptr + outputSize, amount_ptr);
	
	if(stopTimer(&start, &stop, "adj diff", &milliseconds) != cudaSuccess) { goto Error; }
	if(startTimer(&start, &stop) != cudaSuccess) { goto Error; }

	if(handleError(cudaStatus = cudaMemcpy(*outAmount, dev_amount, outputSize * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy dev_amount to host failed")) {
		goto Error;
	}

	if(handleError(cudaStatus = cudaMemcpy(*outText, dev_text, outputSize * sizeof(char), cudaMemcpyDeviceToHost), "cudaMemcpy dev_text to host failed")) {
		goto Error;
	}

	if(stopTimer(&start, &stop, "memcpy", &milliseconds) != cudaSuccess) { goto Error; }

	printf("Total runtime: %f\n", milliseconds);
	printf("%d input, %d output\n", INPUT_SIZE, outputSize);
	
Error:
    cudaFree(dev_in);
    cudaFree(dev_temp);
	cudaFree(dev_amount);
    cudaFree(dev_text);
    return cudaStatus;
}
