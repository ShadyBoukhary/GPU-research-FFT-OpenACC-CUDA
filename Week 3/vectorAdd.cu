// Vector addition in as flattened matrices


#include <stdio.h>


__global__ void matrixMulKernel(float * A_d, float * B_d, float * C_d) {

    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;

    C_d[ty * 64 + tx] = A_d[ty * 64 + tx] + B_d[ty * 64 + tx];
}


int main() {



    float A[4096];
    float B[4096];  
    float C[4096];
    int size = 4096 * sizeof(float);
    float * A_d;
    float * B_d;
    float * C_d;

     for (int x = 0; x < 4096; x++) {
        A[x] = 2.5;
        B[x] = 3.5;
     }
     //printf("%f", A[5]);


    // transfer A, B to device

    cudaMalloc((void **) &A_d, size);
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &B_d, size);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &C_d, size);

    // 1 block of 32x32 = 1024 threads
    dim3 dimGrid0(2, 2 , 1);
    dim3 blockDim0(32, 32, 1);
    matrixMulKernel<<<dimGrid0, blockDim0>>>(A_d, B_d, C_d);





    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

     printf("c[0]=%f\t c[4095]=%f\t", C[0], C[4095]);

    // free memory from device
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

