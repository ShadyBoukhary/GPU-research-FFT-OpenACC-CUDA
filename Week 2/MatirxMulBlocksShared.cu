/*+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
*   Shady Boukhary
*   Midwestern State University
*   CMPS 4563 - Parallel Distributed Computing - GPU Programming
*   February 26th, 2018
*
*
*   CUDA Parallel Code that computes the matrix multiplication of 2 matrices of 512x512 size
*   using 1 block with 1024 threads. The process of multiplication is timed. The resulting
*   matrix and the time it took to be computed are printed to an output file. The GPU Code uses
*   shared memory to speed up performance
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+*/


#include <stdio.h>
#include "timer.h"
#define TILE 32
const int N = 32;

/*+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
*   matrixMuKernell()
*   GPU (device) kernel
*   @param: int *, int *, int *
*   @return: void
*   Description: multiplies 2 matrices and stores result in 3rd matrix using shared mem
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-*/
__global__ void matrixMulKernel(int * A_d, int * B_d, int * C_d, int width) {

    __shared__ int Ads[TILE][TILE];
    __shared__ int Bds[TILE][TILE];

    int tx = threadIdx.x + TILE * blockIdx.x;
    int ty = threadIdx.y + TILE * blockIdx.y;

    int PValue = 0;
    for (int m = 0; m < width/TILE; m++) {
        // load A_d and B_d tiles into shared memory
        Ads[threadIdx.y][threadIdx.x] = A_d[ty * width + m * TILE + threadIdx.x];
        Bds[threadIdx.y][threadIdx.x]= B_d[(m * TILE + threadIdx.y) * width + tx];
        __syncthreads();
        for (int k = 0; k < TILE; k++) {
            PValue += Ads[threadIdx.y][k] * Bds[k][threadIdx.x];
        }
        __syncthreads();
        

    }
    C_d[ty * TILE + tx] = PValue;
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
            B[x][y] = (N-1) - x;
            C[x][y] = 0;
        }
    }
    outfile = fopen("ShadyBoukhary1BOutputS.txt", "w");
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
            fprintf(outfile, "%10i%s", matrix[x][y], " ");
        }
        fprintf(outfile, "\n");
    }
    fprintf(outfile, "\n");

}
