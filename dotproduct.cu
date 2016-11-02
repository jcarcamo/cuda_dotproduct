/*
 * blockAndThread.cu
 * includes setup funtion called from "driver" program
 * also includes kernel function 'cu_fillArray()'
 */

#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 1024

__global__ void kernel_dotproduct(long long *force_d, long long *distance_d, long long *result_d, long long size) {
    extern __shared__ long long sadata[];
    
    int n = blockDim.x;
    int nTotalThreads;
    if (!n){
	nTotalThreads = n;
    }else{
	//(0 == 2^0)
    	int x = 1;
    	while(x < n)
    	{
      	    x <<= 1;
    	}
        nTotalThreads = x;
    }

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    long long i = blockIdx.x*nTotalThreads + threadIdx.x;
    if(i < size){
    	sadata[tid] = force_d[i]*distance_d[i];
    }
    __syncthreads();
    
    // do reduction in shared mem
    //if(i < size){
    for (unsigned int s=1; s < nTotalThreads; s *= 2) {
        if (tid % (2*s) == 0 && (tid+s) < size) {
            sadata[tid] += sadata[tid + s];
        }
        __syncthreads();
    }
    //}  
    // write result for this block to global mem
    if (tid == 0) result_d[blockIdx.x] = sadata[0];
}

// The __global__ directive identifies this function as a kernel
// Note: all kernels must be declared with return type void 
__global__ void kernel_check_threads (long long *force_d, long long *distance_d)
{
    long long x;

    // Note: CUDA contains several built-in variables
    // blockIdx.x returns the blockId in the x dimension
    // threadIdx.x returns the threadId in the x dimension
    x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    force_d[x] = blockIdx.x;
    distance_d[x] = threadIdx.x;
}


// This function is called from the host computer.
// It manages memory and calls the function that is executed on the GPU
extern "C" void cuda_dotproduct (long long *force, long long *distance, long long arraySize, long long *result_array, long long *result)
{
	// block_d and thread_d are the GPU counterparts of the arrays that exists in host memory 
	long long *force_d;
	long long *distance_d;
	long long *result_d;

	cudaError_t op_result;
	
	// allocate space in the device 
	op_result = cudaMalloc ((void**) &force_d, sizeof(long long) * arraySize);
	if (op_result != cudaSuccess) {
		fprintf(stderr, "cudaMalloc (foce) failed.");
		exit(1);
	}
	op_result = cudaMalloc ((void**) &distance_d, sizeof(long long) * arraySize);
	if (op_result != cudaSuccess) {
		fprintf(stderr, "cudaMalloc (distance) failed.");
		exit(1);
	}
	op_result = cudaMalloc ((void**) &result_d, sizeof(long long)*arraySize);
        if (op_result != cudaSuccess) {
                fprintf(stderr, "cudaMalloc (result) failed.");
                exit(1);
        }
	
	//copy the arrays from host to the device 
	op_result = cudaMemcpy (force_d, force, sizeof(long long) * arraySize, cudaMemcpyHostToDevice);
	if (op_result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy host->dev (force) failed.");
		exit(1);
	}
	op_result = cudaMemcpy (distance_d, distance, sizeof(long long) * arraySize, cudaMemcpyHostToDevice);
	if (op_result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy host->dev (distance) failed.");
		exit(1);
	}
        
	op_result = cudaMemcpy (result_d, result_array, sizeof(long long) * arraySize, cudaMemcpyHostToDevice);
        if (op_result != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy host->dev (result) failed.");
                exit(1);
        }
	
        long long blocks = ceil(arraySize / ((float) BLOCK_SIZE));
	// set execution configuration
        dim3 dimblock (BLOCK_SIZE);
        dim3 dimgrid (blocks);
        int smemSize = dimblock.x * sizeof(long long);
        // actual computation: Call the kernel
	kernel_dotproduct<<<dimgrid,dimblock,smemSize>>>(force_d, distance_d, result_d, arraySize);
        //kernel_check_threads<<<dimgrid,dimblock>>>(force_d, distance_d);
        // transfer results back to host
	op_result = cudaMemcpy (force, force_d, sizeof(long long) * arraySize, cudaMemcpyDeviceToHost);
	if (op_result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy host <- dev (force) failed.");
		exit(1);
	}
	op_result = cudaMemcpy (distance, distance_d, sizeof(long long) * arraySize, cudaMemcpyDeviceToHost);
	if (op_result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy host <- dev (distance) failed.");
		exit(1);
	}

	op_result = cudaMemcpy (result_array, result_d, sizeof(long long)*arraySize, cudaMemcpyDeviceToHost);
        if (op_result != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy host <- dev (result) failed.");
                exit(1);
        }

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
        
}

