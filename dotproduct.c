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
#define SIZEOFARRAY 64 

// The serial function dotproduct
void serial_dotproduct(int *force, int *distance, int size, int *result)
{
    int i;
    *result = 0;
    for (i = 0; i < size; i++)
        *result += force[i]*distance[i];
}

// The function dotproduct is in the file dotproduct.cu
extern void cuda_dotproduct(int *force, int *distance, int size, int *result);

int main (int argc, char *argv[])
{
    // Declare arrays and initialize to 0
    int force[SIZEOFARRAY];
    int distance[SIZEOFARRAY];

    // Here's where I could setup the arrays.
    int i;
    int j = 0;
    int k = 0;
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
    int serial_result = 0;
    serial_dotproduct(force, distance, SIZEOFARRAY, &serial_result);
    printf ("Serial dotproduct = %d \n", serial_result);

    int cuda_result = 0;
    // Call the function that will call the GPU function
    cuda_dotproduct (force, distance, SIZEOFARRAY, &cuda_result);

    // Again, print the arrays
    printf ("Final state of the force array:\n");
    for (i=0; i < SIZEOFARRAY; i++) {
        printf ("%d ", force[i]);
    }
    printf ("\n");
    printf ("Final state of the distance array:\n");
    for (i=0; i < SIZEOFARRAY; i++) {
        printf ("%d ", distance[i]);
    }
    printf ("\n");

    return 0;
}
