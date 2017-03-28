
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "output.h"
cudaError_t runLengthEncoding(int *outText, int *outAmount, int *temp, const int *in, unsigned int size);

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

		__syncthreads();
		tmpArr[i + offset] = val;
	}

}
__global__ void encoding(int *outamount, int *outtext, int *prefixSum, const int *text, int outSize, int inSize)
{
	int i = threadIdx.x;
	if(i == 0 || prefixSum[i] > prefixSum[i - 1])
	{
		outtext[prefixSum[i] - 1] = text[i];
		outamount[prefixSum[i] - 1] = i;
		__syncthreads();
		int amount = 0;
		if(prefixSum[i] < outSize)
		{
			amount = outamount[prefixSum[i]] - outamount[prefixSum[i] - 1];
		}
		else
		{
			amount = inSize - outamount[prefixSum[i] - 1]; 
		}
		__syncthreads();
		outamount[prefixSum[i] - 1] = amount;
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
int main()
{
    const int arraySize = 16;
    const int input[arraySize] = { 1, 1, 1, 4, 4, 4, 4, 5, 5, 5, 6, 1, 1, 2, 2, 3 };
	int temp[arraySize] = { 0 };
	//output out = {0, 0};
	int outText[7] = {0, 0, 0, 0, 0, 0, 0}, outAmount[7] = {0, 0, 0, 0, 0, 0, 0};
    // Add vectors in parallel.
	cudaError_t cudaStatus = runLengthEncoding(outText, outAmount, temp, input, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
	
	printArray("Input: ", (int*)input, arraySize);
	printArray("Prefix sum: ", temp, arraySize);
	printArray("Values in input: ", outText, 7);
	printArray("Amount of consecutive values: ", outAmount, 7);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	//free(outText);
	//free(outAmount);
	//outputDispose(out);
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t runLengthEncoding(int *outText, int *outAmount, int *temp, const int *in, unsigned int size)
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
    cudaStatus = cudaMemcpy(dev_in, in, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_temp, temp, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    prefixSum<<<1, size>>>(dev_temp, dev_in, size);
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
    cudaStatus = cudaMemcpy(temp, dev_temp, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	//printArray(temp, size);
	//printArray(dev_temp, size);
	//find size of output array
	//int outputSize = temp[size - 1];
	//outputInit(out, outputSize);


	cudaStatus = cudaMalloc((void**)&dev_amount, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	cudaStatus = cudaMalloc((void**)&dev_text, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	encoding<<<1, size>>>(dev_amount, dev_text, dev_temp, dev_in, 7, size); 
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

	cudaStatus = cudaMemcpy(outAmount, dev_amount, 7 * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	cudaStatus = cudaMemcpy(outText, dev_text, 7 * sizeof(int), cudaMemcpyDeviceToHost);
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
