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

const int N = 512;

/*+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
*   matrixMuKernell()
*   GPU (device) kernel
*   @param: int *, int *, int *
*   @return: void
*   Description: multiplies 2 matrices and stores result in 3rd matrix
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-*/
__global__ void matrixMulKernel(int * A_d, int * B_d, int * C_d, int width) {
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;

    int PValue = 0;
    for (int x = 0; x < width; x++){
        int aElement = A_d[ty * width + x];
        int bElement = B_d[x * width + tx];
        PValue += aElement * bElement;
    }
    C_d[ty * width + tx] = PValue;
}


void printMatrix(int [][N], FILE*);
int main() {


    FILE *outfile;
    double timeStart, timeStop, timeElapsed;


    int A[N][N], B[N][N], C[N][N];
    int size = N * N * sizeof(int);
    int * A_d;
    int * B_d;
    int * C_d;

    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++){
            A[x][y] = x;
            B[x][y] = 511 - x;
            C[x][y] = 2;
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
    dim3 dimGrid0(16, 16 , 1);
    dim3 blockDim0(32, 32, 1);

    GET_TIME(timeStart);
    matrixMulKernel<<<dimGrid0, blockDim0>>>(A_d, B_d, C_d, N);
    cudaDeviceSynchronize();
    GET_TIME(timeStop);

    timeElapsed = timeStop - timeStart;




    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    printMatrix(C, outfile);

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
void printMatrix(int matrix[][N], FILE* outfile){
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++){
            fprintf(outfile, "%8i%s", matrix[x][y], " ");
        }
        fprintf(outfile, "\n");
    }
    fprintf(outfile, "\n");

}
