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

template <unsigned int blockSize>
__global__ void kernel_dotproduct2(long long *force_d, long long *distance_d, long long *result_d, long long size)
{
	extern __shared__ long long sdata[];
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

	unsigned int tid = threadIdx.x;
	long long i = blockIdx.x*(nTotalThreads*2) + threadIdx.x;
	if((i+nTotalThreads)< size){
		sdata[tid] = force_d[i]*distance_d[i] + force_d[i+nTotalThreads]*distance_d[i+nTotalThreads] ;
	} else {
		if(i < size){
			sdata[tid] = force_d[i]*distance_d[i];
		}else{
			sdata[tid] = 0;
		}
	}
	__syncthreads();
	for (long long s=nTotalThreads/2; s>32 && (tid+s) < size; s>>=1)
	{
		if (tid < s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	if (tid < 32)
	{
		sdata[tid] += sdata[tid + 32];
		sdata[tid] += sdata[tid + 16];
		sdata[tid] += sdata[tid + 8];
		sdata[tid] += sdata[tid + 4];
		sdata[tid] += sdata[tid + 2];
		sdata[tid] += sdata[tid + 1];
	}

	// write result for this block to global mem
	if (tid == 0) result_d[blockIdx.x] = sdata[0];
	
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
	
	int threads;
	if (arraySize < 256 ){
		threads = 128;
	} else if (arraySize < 512){
		threads = 256;
	} else if (arraySize < 1024){
		threads = 512;
	} else {
		threads = BLOCK_SIZE;
	}
	long long block_size = threads;
        long long blocks = ceil(arraySize / ((float) block_size));
	// set execution configuration
        dim3 dimblock (block_size);
        dim3 dimgrid (blocks);
        int smemSize = dimblock.x * sizeof(long long);
        // actual computation: Call the kernel
	//kernel_dotproduct<<<dimgrid,dimblock,smemSize>>>(force_d, distance_d, result_d, arraySize);
        switch (threads)
	{
		case 128:
		  kernel_dotproduct2<128><<<dimgrid,dimblock,smemSize>>>(force_d, distance_d, result_d, arraySize);
		  break;
		case 256:
                  kernel_dotproduct2<256><<<dimgrid,dimblock,smemSize>>>(force_d, distance_d, result_d, arraySize);
                  break;
		case 512:
                  kernel_dotproduct2<256><<<dimgrid,dimblock,smemSize>>>(force_d, distance_d, result_d, arraySize);
                  break;
		default:
		 kernel_dotproduct2<BLOCK_SIZE><<<dimgrid,dimblock,smemSize>>>(force_d, distance_d, result_d, arraySize); 
		 break;
	}
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
	
	int i,j = 0;
	for (i = 0; i < arraySize; i++){
		if(result_array[i] < 0){
			j++;
		}
	}
	printf("faulty # = %d \n",j);

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

