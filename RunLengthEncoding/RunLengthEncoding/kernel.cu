
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
#include <thrust/iterator/reverse_iterator.h>
#include "output.h"

//#define INPUT_SIZE 73728
#define INPUT_SIZE 39942400
cudaError_t runLengthEncoding(char **outText, int **outAmount, int **temp, char **in, unsigned int size, int *outSize);
//example input: a, b, b, c, c, c, d, e, e
//flags:		 1, 1, 0, 1, 0, 0, 1, 1, 0
//prefix sum:    1, 2, 2, 3, 3, 3, 4, 5, 5
__global__ void prefixSum(int *tmpArr, const char *text, int size)
{
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int i = blockId * blockDim.x + threadIdx.x;
    if(i == 0 || text[i] != text[i - 1])
	{
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
__global__ void encoding(int *outAmount, char *outText, int *prefixSum, const char *text, int outSize, int inSize)
{
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int i = blockId * blockDim.x + threadIdx.x;
	if(i == 0 || prefixSum[i] > prefixSum[i - 1])
	{
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
	for(int i = 0; i < size; i++)
	{
		printf("%d, ", arr[i]);
	}
	putchar('\n');
}

void printCharArray(char *msg, char *arr, int size)
{
	printf("%s: ", msg);
	for(int i = 0; i < size; i++)
	{
		printf("%c, ", arr[i]);
	}
	putchar('\n');
}

int initializeArray(int **arr, int size, bool fillRandom)
{
	*arr = (int*)malloc(size * sizeof(int));

	if(*arr == NULL)
	{
		return 1;
	}
	if(fillRandom)
	{
		srand(NULL);
		for(int i = 0; i < size; i++)
		{
			(*arr)[i] = rand() % 5 + 63;
		}
	}
	else
	{
		for(int i = 0; i < size; i++)
		{
			(*arr)[i] = 0;
		}
		return 0;
	}
	return 0;
}

int initializeCharArray(char **arr, int size, bool fillRandom)
{
	*arr = (char*)malloc(size * sizeof(char));

	if(*arr == NULL)
	{
		return 1;
	}
	if(fillRandom)
	{
		srand(NULL);
		for(int i = 0; i < size; i++)
		{
			(*arr)[i] = (char)(rand() % 5 + 'A');
		}
		(*arr)[size - 1] = (char)(((*arr)[size - 2] + 32) % 256);
	}
	else
	{
		for(int i = 0; i < size; i++)
		{
			(*arr)[i] = 0;
		}
		return 0;
	}
	return 0;
}

void checkOutput(char *input, char *outText, int *outAmount, int outputSize)
{
	for(int i = 0, j = 0; i < outputSize && j < INPUT_SIZE; i++)
	{
		if(input[j] == outText[i])
		{
			int count = j + outAmount[i + 1];
			while(j < count)
			{
				//printf("Input: %c, Output: %c, Amount: %d\n", input[j], outText[i], outAmount[i + 1]);
				if(input[j] != outText[i])
				{
					printf("Error at %d in outAmount\n", i + 1);
				}
				j++;
			}
		}
		else
		{
			printf("Error at %d in outText\n", i);
		}
	}
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
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, prop.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, prop.maxGridSize[i]);
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
	cudaError_t cudaStatus = runLengthEncoding(&outText, &outAmount, &prefix, &input, arraySize, pOutSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
	
	/*printCharArray("Input: ", input, arraySize);
	printArray("Prefix sum: ", prefix, arraySize);
	printCharArray("Values in input: ", outText, outSize);
	printArray("Amount of consecutive values: ", outAmount, outSize);*/
	checkOutput(input, outText, outAmount, outSize);
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
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
	dim3 gridSize(395, 395, 1);
	dim3 blockSize(256, 1, 1);
	float milliseconds = 0;

    cudaError_t cudaStatus;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_in, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	cudaStatus = cudaMalloc((void**)&dev_temp, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_in, *in, size * sizeof(char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_temp, *temp, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f elapsed time : memcpy\n", milliseconds);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
    // Launch a kernel on the GPU with one thread for each element.
	prefixSum<<<gridSize, blockSize>>>(dev_temp, dev_in, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "prefixSum launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching prefixSum!\n", cudaStatus);
        goto Error;
    }
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f elapsed time : flags\n", milliseconds);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	thrust::device_ptr<int> temp_ptr = thrust::device_pointer_cast<int>(dev_temp);
	thrust::inclusive_scan(thrust::device, temp_ptr, temp_ptr + size, temp_ptr);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f elapsed time : scan\n", milliseconds);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
     //Copy size of output from GPU to memory.
    cudaStatus = cudaMemcpy(*temp + size - 1, dev_temp + size - 1, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f elapsed time : memcpy\n", milliseconds);
	//find size of output array
	int outputSize = (*temp)[size - 1];

	(*temp)[size - 1] = INPUT_SIZE;
	*outSize = outputSize;
	if(initializeCharArray(outText, outputSize, false)) { printf("Error malloc outText %d outputSize\n", outputSize); goto Error; }
	if(initializeArray(outAmount, outputSize, false)) { printf("Error malloc outAmount %d outputSize\n", outputSize); goto Error; }

	cudaStatus = cudaMalloc((void**)&dev_amount, outputSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	cudaStatus = cudaMalloc((void**)&dev_text, outputSize * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	encoding<<<gridSize, blockSize>>>(dev_amount, dev_text, dev_temp, dev_in, outputSize, size);

	cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "encoding launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching encoding!\n", cudaStatus);
        goto Error;
    }
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f elapsed time : encoding\n", milliseconds);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	thrust::device_ptr<int> amount_ptr = thrust::device_pointer_cast<int>(dev_amount);
	thrust::adjacent_difference(thrust::device, amount_ptr, amount_ptr + outputSize, amount_ptr);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f elapsed time : adj diff\n", milliseconds);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	cudaStatus = cudaMemcpy(*outAmount, dev_amount, outputSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	cudaStatus = cudaMemcpy(*outText, dev_text, outputSize * sizeof(char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f elapsed time : memcpy\n", milliseconds);
	printf("%d input, %d output\n", INPUT_SIZE, outputSize);
Error:
    cudaFree(dev_in);
    cudaFree(dev_temp);
	cudaFree(dev_amount);
    cudaFree(dev_text);
    return cudaStatus;
}
