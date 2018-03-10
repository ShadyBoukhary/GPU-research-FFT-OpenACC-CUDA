 #include <stdio.h>
 #include <stdlib.h>
 #include "timer.h"

 int main(){

    int n; /* vector length */
    double start, stop, elapsed;

    float * a; /* input vector 1 */
    float * b; /* input vector 2 */
    float * r; /* output vector */
    float * e; /* expected output values */
    //if( argc > 1 ) n = atoi( argv[1] );
     /* default vector length */
    n = 8192;
    a = (float*)malloc( n*n*sizeof(float) );
    b = (float*)malloc( n*n*sizeof(float) );
    r = (float*)malloc( n*n*sizeof(float) );
    e = (float*)malloc( n*n*sizeof(float) );

     /* compute on the GPU */
     
    float h = 0;
    for(int i = 0; i < n; ++i ){
        for (int j = 0; j < n; j++) {
            a[i*n+j] = (float)(++h);
            b[i*n+j] = (float)(n * n - h);
        }
     }
    GET_TIME(start);

     // acc directive determines this is an openACC directive
     // data directive tells compiler we want to move data from host to device and vice versa
     // copyin means copy to device, copy a of size n*n starting at index 0
     // after creating kernel, copy r back from device to host
     // pragma kernels tells the compiler to analyze code and determine block and grid sizes
     #pragma acc data copyin(a[0:n*n],b[0:n*n]) copyout(r[0:n*n])
     #pragma acc region
     #pragma acc loop independent vector(32)
     for( int i = 0; i < n; ++i ) {
         #pragma acc loop independent vector(32)
         for(int j = 0; j < n; j++) {
             float sum = 0;
             for(int k = 0; k < n; k++) {
                 sum += a[i * n + k] * b[k * n + j];
             }
             r[i * n + j] = sum;
         }
         
     }
     GET_TIME(stop);
     elapsed = stop - start;
     printf("%e\n", elapsed);



     return 0;

} 