/*
 * blockAndThread.cu
 * includes setup funtion called from "driver" program
 * also includes kernel function 'cu_fillArray()'
 */

#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 64
 
__global__ void kernel_dotproduct(int *force_d, int *distance_d, int *result_d) {
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x
    *blockDim.x + threadIdx.x;
    sdata[tid] = force_d[i]*distance_d[i];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2) {
      if (tid % (2*s) == 0) {
      sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) result_d[blockIdx.x] = sdata[0];
}

/*
template <unsigned int blockSize>
__global__ void reduce6(int *force_d, int *distance_d, int *result_d, unsigned int size)
{
    extern __shared__ int sdata[];
    /*unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    //while (i < size) { sdata[tid] += force_d[i]*distance_d[i] + force_d[i+blockSize]*distance_d[i+blockSize]; i += gridSize; }
    while (i < size) { 
	sdata[tid] += force_d[i] + force_d[i+blockSize];
	i += gridSize;
    }// /
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = force_d[i] + force_d[i+blockDim.x];
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
    }
    if (tid == 0) result_d[blockIdx.x] = sdata[0];
}

// The __global__ directive identifies this function as a kernel
// Note: all kernels must be declared with return type void 
__global__ void kernel_dotproduct (int *force_d, int *distance_d)
{
    int x;

    // Note: CUDA contains several built-in variables
    // blockIdx.x returns the blockId in the x dimension
    // threadIdx.x returns the threadId in the x dimension
    x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	force_d[x] = blockIdx.x;
	distance_d[x] = threadIdx.x;
}
*/

// This function is called from the host computer.
// It manages memory and calls the function that is executed on the GPU
extern "C" void cuda_dotproduct (int *force, int *distance, int arraySize, int *result)
{
	// block_d and thread_d are the GPU counterparts of the arrays that exists in host memory 
	int *force_d;
	int *distance_d;
	int *result_d;
	cudaError_t op_result;

	// allocate space in the device 
	op_result = cudaMalloc ((void**) &force_d, sizeof(int) * arraySize);
	if (op_result != cudaSuccess) {
		fprintf(stderr, "cudaMalloc (foce) failed.");
		exit(1);
	}
	op_result = cudaMalloc ((void**) &distance_d, sizeof(int) * arraySize);
	if (op_result != cudaSuccess) {
		fprintf(stderr, "cudaMalloc (distance) failed.");
		exit(1);
	}
	
	op_result = cudaMalloc ((void**) &result_d, sizeof(int)*arraySize);
        if (op_result != cudaSuccess) {
                fprintf(stderr, "cudaMalloc (distance) failed.");
                exit(1);
        }

	//copy the arrays from host to the device 
	op_result = cudaMemcpy (force_d, force, sizeof(int) * arraySize, cudaMemcpyHostToDevice);
	if (op_result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy host->dev (force) failed.");
		exit(1);
	}
	op_result = cudaMemcpy (distance_d, distance, sizeof(int) * arraySize, cudaMemcpyHostToDevice);
	if (op_result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy host->dev (distance) failed.");
		exit(1);
	}
        
        op_result = cudaMemcpy (result_d, distance, sizeof(int) * arraySize, cudaMemcpyHostToDevice);
        if (op_result != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy host->dev (result) failed.");
                exit(1);
        }
	
	// set execution configuration
	dim3 dimblock (BLOCK_SIZE);
	dim3 dimgrid (arraySize/BLOCK_SIZE);
        int smemSize = dimblock.x * sizeof(int);
	// actual computation: Call the kernel
	kernel_dotproduct<<<dimgrid,dimblock,smemSize>>>(force_d, distance_d, result_d);
	
        // transfer results back to host
	op_result = cudaMemcpy (force, force_d, sizeof(int) * arraySize, cudaMemcpyDeviceToHost);
	if (op_result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy host <- dev (force) failed.");
		exit(1);
	}
	op_result = cudaMemcpy (distance, distance_d, sizeof(int) * arraySize, cudaMemcpyDeviceToHost);
	if (op_result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy host <- dev (distance) failed.");
		exit(1);
	}
	int result_array[arraySize];

	op_result = cudaMemcpy (result_array, result_d, sizeof(int)*arraySize, cudaMemcpyDeviceToHost);
        if (op_result != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy host <- dev (distance) failed.");
                exit(1);
        }
        *result = result_array[0];
	printf("result array: [");
	int re;
	for (re = 0; re < arraySize; re++){
		printf(" %d ", result_array[re]);
	}
	printf("]\n");
	// release the memory on the GPU 
	op_result = cudaFree (force_d);
	if (op_result != cudaSuccess) {
		fprintf(stderr, "cudaFree (force) failed.");
		exit(1);
	}
	op_result = cudaFree (distance_d);
	if (op_result != cudaSuccess) {
		fprintf(stderr, "cudaFree (distance) failed.");
		exit(1);
	}
	op_result = cudaFree (result_d);
        if (op_result != cudaSuccess) {
                fprintf(stderr, "cudaFree (distance) failed.");
                exit(1);
        }
        printf("CUDA result: %d \n", *result);
}

