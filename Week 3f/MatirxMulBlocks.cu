/*+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
*   Shady Boukhary
*   Midwestern State University
*   CMPS 4563 - Parallel Distributed Computing - GPU Programming
*   HW 2 
*   February 19th, 2018
*
*
*   CUDA Parallel Code that computes the matrix multiplication of 2 matrices of 32x32 size
*   using 1 block with 1024 threads. The process of multiplication is timed. The resulting
*   matrix and the time it took to be computed are printed to an output file.
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+*/


#include <stdio.h>
#include "timer.h"

const int N = 8192;

/*+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
*   matrixMuKernell()
*   GPU (device) kernel
*   @param: int *, int *, int *
*   @return: void
*   Description: multiplies 2 matrices and stores result in 3rd matrix
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-*/
__global__ void matrixMulKernel(float * A_d, float * B_d, float * C_d, int width) {
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    float PValue = 0;
    for (int x = 0; x < width; x++){
        float aElement = A_d[ty * width + x];
        float bElement = B_d[x * width + tx];
        PValue += aElement * bElement;
    }
    C_d[ty * width + tx] = PValue;
}


void printMatrix(float* , FILE*);
int main() {


    FILE *outfile;
    double timeStart, timeStop, timeElapsed;


    float *A = (float *)malloc(N * N * sizeof(float)); 
    float *B = (float *)malloc(N * N * sizeof(float)); 
    float *C = (float *)malloc(N * N * sizeof(float)); 
    int size = N * N * sizeof(float);
    float * A_d;
    float * B_d;
    float * C_d;

    float h = 0;
     for (int x = 0; x < N; x++) {
         for (int y = 0; y < N; y++){
             A[x * N + y] = ++h;
             B[x * N + y] = N*N - h;
             C[x * N + y] = 0;
         }
     }
    outfile = fopen("ShadyBoukhary1BOutput.txt", "w");
    if (outfile == NULL) {
        printf("%s", "Failed to open file.\n");
        exit(1);
    }

    // transfer A, B to device

    cudaMalloc((void **) &A_d, size);
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &B_d, size);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &C_d, size);

    // 1 block of 32x32 = 1024 threads
    dim3 dimGrid0(N / 32, N / 32 , 1);
    dim3 blockDim0(32, 32, 1);

    GET_TIME(timeStart);
    matrixMulKernel<<<dimGrid0, blockDim0>>>(A_d, B_d, C_d, N);
    cudaDeviceSynchronize();
    GET_TIME(timeStop);

    timeElapsed = timeStop - timeStart;




    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    //printMatrix(C, outfile);

    fprintf(outfile, "The code to be timed took %e seconds\n", timeElapsed);
    // free memory from device
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

/*+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
*   printMatrix()
*   @param: int[][], int[][], FILE*
*   @return: void
*   Description: prints matrix to an output file
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-*/
void printMatrix(float * matrix, FILE* outfile){
     
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++){
            fprintf(outfile, "%5f%s", matrix[x * N + y], " ");
        }
        fprintf(outfile, "\n");
    }
    fprintf(outfile, "\n");

}