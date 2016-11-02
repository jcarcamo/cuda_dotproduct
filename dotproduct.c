/*
 * blockAndthread.c
 * A "driver" program that calls a routine (i.e. a kernel)
 * that executes on the GPU.  The kernel fills two int arrays
 * with the block ID and the thread ID
 *
 * Note: the kernel code is found in the file 'blockAndThread.cu'
 * compile both driver code and kernel code with nvcc, as in:
 * 			nvcc blockAndThread.c blockAndThread.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SIZEOFARRAY 10240000

struct timeval  ts1, ts2, tp1, tp2, tps1, tps2;
// The serial function dotproduct
void serial_dotproduct(long long *force, long long *distance, long long size, long long *result)
{
    int i;
    *result = 0;
    for (i = 0; i < size; i++)
        *result += force[i]*distance[i];
}

// The function dotproduct is in the file dotproduct.cu
extern void cuda_dotproduct(long long *force, long long *distance, long long size, long long *result_array, long long *result);

int main (int argc, char *argv[])
{
    //timeval tv1, tv2;
    // Declare arrays and initialize to 0
    long long *force;
    force = (long long*)malloc(SIZEOFARRAY*sizeof(long long));
    long long *distance;
    distance = (long long*)malloc(SIZEOFARRAY*sizeof(long long));
    long long *result_array;
    result_array = (long long*)malloc(SIZEOFARRAY*sizeof(long long));
    gettimeofday(&tps1, NULL);
    // Here's where I could setup the arrays.
    long long i;
    long long j = 0;
    long long k = 0;
    for (i=0; i < SIZEOFARRAY; i++) {
        if(i < SIZEOFARRAY/2){
            force[i]=i+1;
	}else{
	    force[i]=SIZEOFARRAY/2-j;
	    j++;
	}
	if (i%10 != 0){
	    distance[i]=k+1;
            k++;
	}else{
	    distance[i]=1;
	    k = 1;
	}
    }
    for(i=0;i<SIZEOFARRAY;i++){
        result_array[i]=0;
    }
    gettimeofday(&tps2, NULL);
    double tps_time = (double) (tps2.tv_usec - tps1.tv_usec) / 1000000 + (double) (tps2.tv_sec - tps1.tv_sec);
    printf("tps: %f\n",tps_time);
    
    // Print the initial arrays
    /*
    printf ("Initial state of the force array:\n");
    for (i=0; i < SIZEOFARRAY; i++) {
        printf ("%d ", force[i]);
    }
    printf ("\n");
    printf ("Initial state of the distance array:\n");
    for (i=0; i < SIZEOFARRAY; i++) {
        printf ("%d ", distance[i]);
    }
    printf ("\n");
    */

    // Serial dotproduct
    long long serial_result = 0;
    gettimeofday(&ts1, NULL);
    serial_dotproduct(force, distance, SIZEOFARRAY, &serial_result);
    gettimeofday(&ts2, NULL);
    double ts_time = (double) (ts2.tv_usec - ts1.tv_usec) / 1000000 + (double) (ts2.tv_sec - ts1.tv_sec);
    printf("ts: %f\n",ts_time);
    printf ("Serial dotproduct = %lld \n", serial_result);
    
    long long cuda_result = 0; 
    gettimeofday(&tp1, NULL);
    // Call the function that will call the GPU function
    cuda_dotproduct (force, distance, SIZEOFARRAY, result_array, &cuda_result);
    gettimeofday(&tp2, NULL);
    double tp_time = (double) (tp2.tv_usec - tp1.tv_usec) / 1000000 + (double) (tp2.tv_sec - tp1.tv_sec);
    printf("tp: %f\n",tp_time);
    //printf("result array: [");
    for (i = 0; i < SIZEOFARRAY; i++){
        //printf(" %d ", result_array[i]);
        /*if(result_array[i] != 0){
            printf(" %d ", result_array[i]);
        }*/
        cuda_result += result_array[i];
    }
    //printf("]\n");
    printf("CUDA result: %lld \n",cuda_result);
    // Again, print the arrays
    /*printf ("Final state of the force array:\n");
    for (i=0; i < SIZEOFARRAY; i++) {
        printf ("%d ", force[i]);
    }
    printf ("\n");
    printf ("Final state of the distance array:\n");
    for (i=0; i < SIZEOFARRAY; i++) {
        printf ("%d ", distance[i]);
    }
    printf ("\n");
    */
    return 0;
}
