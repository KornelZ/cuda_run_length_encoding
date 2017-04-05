
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "output.h"

#define INPUT_SIZE 64

cudaError_t runLengthEncoding(int **outText, int **outAmount, int **temp, int **in, unsigned int size, int *outSize);
//example input: a, b, b, c, c, c, d, e, e
//flags:		 1, 1, 0, 1, 0, 0, 1, 1, 0
//prefix sum:    1, 2, 2, 3, 3, 3, 4, 5, 5
__global__ void prefixSum(int *tmpArr, const int *text, int size)
{
	int i = threadIdx.x, logSize = logf(size);
    if(i == 0 || text[i] != text[i - 1])
	{
		tmpArr[i] = 1;
	}

	//prefix sum(naive)
	int val = 0;
	for(int j = 0, offset = 1; j <= logSize + 1; j++, offset *= 2)
	{
		__syncthreads();
		if(i + offset < size)
		{
			val = tmpArr[i] + tmpArr[i + offset];
		}
		tmpArr[i + offset] = val;
	}

}
//i:             0, 1, 2, 3, 4, 5, 6, 7, 8
//example input: a, b, b, c, c, c, d, e, e
//prefix sum:    1, 2, 2, 3, 3, 3, 4, 5, 5
//index:         0, 1, -, 2, -, -, 3, 4, -
//outText:       a, b, c, d, e
//Amount I:      0, 1, 3, 6, 7 --> 1 - 0, 3 - 1, 6 - 3, 7 - 6, (size) - 7
//Amount II:     1, 2, 3, 1, 2 --> 1,     2,   , 3    , 1    , 2
__global__ void encoding(int *outAmount, int *outText, int *prefixSum, const int *text, int outSize, int inSize)
{
	int i = threadIdx.x;
	if(i == 0 || prefixSum[i] > prefixSum[i - 1])
	{
		int index = prefixSum[i] - 1;
		outText[index] = text[i];
		outAmount[index] = i;
		__syncthreads();
		int amount = 0;
		if(prefixSum[i] < outSize)
		{
			amount = outAmount[prefixSum[i]] - outAmount[index];
		}
		else
		{
			amount = inSize - outAmount[index]; 
		}
		__syncthreads();
		outAmount[index] = amount;
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
			(*arr)[i] = rand() % 5;
		}
	}
	else
	{
		for(int i = 0; i < size; i++)
		{
			(*arr)[i] = 0;
		}
	}
	return 0;
}

void checkOutput(int *input, int *outText, int *outAmount, int outputSize)
{
	for(int i = 0, j = 0; i < outputSize && j < INPUT_SIZE; i++)
	{
		if(input[j] == outText[i])
		{
			int count = j + outAmount[i];
			while(j < count)
			{
				if(input[j] != outText[i])
				{
					printf("Error at %d in outAmount\n", i);
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

int main()
{
	int arraySize = INPUT_SIZE;
    int *input = 0;
	int *prefix = 0;

	int *outText = 0, *outAmount = 0;
	int outSize = 0;
	int *pOutSize = &outSize;
	if(initializeArray(&input, arraySize, true)) { printf("Error malloc input\n"); return; }
	if(initializeArray(&prefix, arraySize, false)) { printf("Error malloc input\n"); return; }
	cudaError_t cudaStatus = runLengthEncoding(&outText, &outAmount, &prefix, &input, arraySize, pOutSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
	
	printArray("Input: ", input, arraySize);
	printArray("Prefix sum: ", prefix, arraySize);
	printArray("Values in input: ", outText, outSize);
	printArray("Amount of consecutive values: ", outAmount, outSize);
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
cudaError_t runLengthEncoding(int **outText, int **outAmount, int **temp, int **in, unsigned int size, int *outSize)
{
    int *dev_in = 0;
	int *dev_temp = 0;
	int *dev_amount = 0, *dev_text = 0;

    cudaError_t cudaStatus;

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
    cudaStatus = cudaMemcpy(dev_in, *in, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_temp, *temp, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    prefixSum<<<1, 64>>>(dev_temp, dev_in, size);
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

     //Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(*temp, dev_temp, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	//find size of output array
	int outputSize = (*temp)[size - 1];
	*outSize = outputSize;
	if(initializeArray(outText, outputSize, false)) { printf("Error malloc outText\n"); goto Error; }
	if(initializeArray(outAmount, outputSize, false)) { printf("Error malloc outAmount\n"); goto Error; }

	cudaStatus = cudaMalloc((void**)&dev_amount, outputSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	cudaStatus = cudaMalloc((void**)&dev_text, outputSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	encoding<<<1, size>>>(dev_amount, dev_text, dev_temp, dev_in, outputSize, size); 
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

	cudaStatus = cudaMemcpy(*outAmount, dev_amount, outputSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	cudaStatus = cudaMemcpy(*outText, dev_text, outputSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
Error:
    cudaFree(dev_in);
    cudaFree(dev_temp);
	cudaFree(dev_amount);
    cudaFree(dev_text);
    return cudaStatus;
}
