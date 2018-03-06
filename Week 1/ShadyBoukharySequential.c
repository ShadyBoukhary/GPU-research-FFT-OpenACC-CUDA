/*+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
*   Shady Boukhary
*   Midwestern State University
*   CMPS 4563 - Parallel Distributed Computing - GPU Programming
*   HW 2 
*   February 19th, 2018
*
*
*   Sequential Code that computes the matrix multiplication of 2 matrices of 32x32 size
*   The process of multiplication is timed. The resulting
*   matrix and the time it took to be computed are printed to an output file.
*
*
*   gcc -std=c99 -o a.out ShadyBoukharySequential.c
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+*/


#include <stdio.h>
#include "timer.h"

const int N = 32;

void matrixMul(int [][N], int [][N], int [][N], int);
void printMatrix(int [][N], FILE*);


int main() {


    FILE *outfile;
    double timeStart, timeStop, timeElapsed;


    int A[N][N], B[N][N];
    int C[N][N];

    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++){
            A[x][y] = x;
            B[x][y] = 31 - x;
            C[x][y] = 0;
        }
    }
    outfile = fopen("ShadyBoukharySequentialOutput.txt", "w");
    if (outfile == NULL) {
        printf("%s", "Failed to open file.\n");
        exit(1);
    }

    
    GET_TIME(timeStart);
    matrixMul(A, B, C, N);
    GET_TIME(timeStop);

    timeElapsed = timeStop - timeStart;
    printMatrix(C, outfile);



    fprintf(outfile, "The code to be timed took %e seconds\n", timeElapsed);

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
            fprintf(outfile, "%5i%s", matrix[x][y], " ");
        }
        fprintf(outfile, "\n");
    }
    fprintf(outfile, "\n");

}

/*+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
*   matrixMul()
*   @param: int[][], int[][], int[][]
*   @return: void
*   Description: multiplies 2 matrices and stores result in 3rd matrix
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-*/
void matrixMul(int M[][N], int B[][N], int C[][N], int width) {
    int sum = 0;
    for (int i = 0 ; i < width ; i++ ){
      for (int j = 0 ; j < width ; j++ ){
        for (int k = 0 ; k < width ; k++ ){
          sum = sum + M[i][k] * B[k][j];
        }
 
        C[i][j] = sum;
        sum = 0;
      }
    }

}
