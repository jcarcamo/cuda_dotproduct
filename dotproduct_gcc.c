#include <stdio.h>
#include <stdlib.h>

// The serial function dotproduct
void serial_dotproduct(int *force, int *distance, int size, long long *result)
{
    int i;
    *result = 0;
    for (i = 0; i < size; i++)
        *result += force[i]*distance[i];
}

int main (int argc, char *argv[])
{
    //timeval tv1, tv2;
    int SIZEOFARRAY = 2000000;
    printf("First step");
    // Declare arrays and initialize to 0
    int *force;
    force = (int*)malloc(SIZEOFARRAY*sizeof(int));
    int *distance;
    distance = (int*)malloc(SIZEOFARRAY*sizeof(int));

    //gettimeofday(&tv1, NULL);
    // Here's where I could setup the arrays.
    int i;
    int j = 0;
    int k = 0;
    for (i=0; i < SIZEOFARRAY; i++) {
        if(i < SIZEOFARRAY/2){
            force[i]=i+1;
	    printf("%d ",i);
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
    /*printf ("Initial state of the force array:\n");
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
    //gettimeofday(&tv1, NULL);
    serial_dotproduct(force, distance, SIZEOFARRAY, &serial_result);
    printf ("Serial dotproduct = %lld \n", serial_result);

    return 0;
}
