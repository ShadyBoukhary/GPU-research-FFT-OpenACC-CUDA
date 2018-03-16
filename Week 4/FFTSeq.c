/*+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
*   Shady Boukhary
*   Midwestern State University
*   Research Week 4 - Fast Fourier Transform: Cooley Tukey
*   March 15th, 2018
*
*
*   Computes a radix-2 fast fourier transform using an iterative implementation of
*	the Cooley-Tukey Algorithm
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+*/

#include <stdio.h>      
#include <complex.h>    
#include <math.h>

#define PI 3.14159265
#define SIZE 8


_Dcomplex * computeFFT(_Dcomplex *, int);
_Dcomplex eIThetta(int, int, int, int);
_Dcomplex multComplex(_Dcomplex, _Dcomplex);
_Dcomplex addComplex(_Dcomplex, _Dcomplex);
void printFFT(const _Dcomplex *, int);

int main()
{
	_Dcomplex * signals = (_Dcomplex*)malloc(SIZE * sizeof(_Dcomplex));
	_Dcomplex * fft = (_Dcomplex*)malloc(SIZE * sizeof(_Dcomplex));
	_Dcomplex comp = { 3.6, 2.6 };
	_Dcomplex comp1 = { 2.9, 6.3 };
	_Dcomplex comp2 = { 5.6, 4.0 };
	_Dcomplex comp3 = { 4.8, 9.1 };
	_Dcomplex comp4 = { 3.3, 0.4 };
	_Dcomplex comp5 = { 5.9, 4.8 };
	_Dcomplex comp6 = { 5.0, 2.6 };
	_Dcomplex comp7 = { 4.3, 4.1 };

	signals[0] = comp;;
	signals[1] = comp1;
	signals[2] = comp2;
	signals[3] = comp3;
	signals[4] = comp4;
	signals[5] = comp5;
	signals[6] = comp6;
	signals[7] = comp7;

	
	fft = computeFFT(signals, SIZE);
	printFFT(fft, SIZE);
	getchar();

}

/*+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
*   computeFFT()
*   @param: _Dcomplex, int
*   @return: _Dcomplex
*   Description: computes FFT for a signal
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-*/
_Dcomplex * computeFFT(_Dcomplex * numbers, int N) {

	/* X: energy
		k: frequency
		n: signal
		eIThetta: spin of the signal at different angles
		
		Basically calculates the energy at specific frequencies by spinning
		the signal around a circle at that frequency and summing up various points
		around that circle															*/


	// array to hold X(k)
	_Dcomplex * X = (_Dcomplex*)malloc(N * sizeof(_Dcomplex));

	// compute all X(K)
	for (int k = 0; k < N; k++) {
		_Dcomplex sumEven = { 0.0, 0.0 };
		_Dcomplex sumOdd = { 0.0, 0.0 };

		for (int n = 0; n <= (N / 2) - 1; n++) {
			// compute the even part

			_Dcomplex comp = numbers[2 * n];
			_Dcomplex eThetta = eIThetta(k, N, n, 0);
			_Dcomplex resultEven = multComplex(comp, eThetta);
			sumEven = addComplex(resultEven, sumEven);

			// compute the odd part

			_Dcomplex compOdd = numbers[2 * n + 1];
			_Dcomplex eThettaOdd = eIThetta(k, N, n, 1);
			_Dcomplex resultOdd = multComplex(compOdd, eThettaOdd);
			sumOdd = addComplex(resultOdd, sumOdd);
		}
		// compute X(k)
		X[k] = addComplex(sumEven, sumOdd);
	}
	return X;
}

/*+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
*   eIThetta()
*   @param: int, int, int, int
*   @return: _Dcomplex
*   Description: computes the spin of the signal around a circle at its frequency
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-*/
_Dcomplex eIThetta(int k, int N, int n, int offset) {
	// compute real part
	double realPart = cos((2 * PI * (2 * n + offset) * k) / N);

	// compute imaginary part
	double imaginaryPart = (-1) * sin((2 * PI * (2 * n + offset) * k) / N);

	// create a complex number out of them and return it
	_Dcomplex result = { realPart, imaginaryPart };
	return result;
}


/*+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
*   multComplex()
*   @param: _Dcomplex, _Dcomplex
*   @return: _Dcomplex
*   Description: multiplies 2 complex numbers
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-*/
_Dcomplex multComplex(_Dcomplex a, _Dcomplex b) {
	// complex number multiplication
	double realPart = (creal(a) * creal(b)) - (cimag(a) * cimag(b));
	double imagPart = (creal(a) * cimag(b)) + (cimag(a) * creal(b));

	_Dcomplex result = { realPart, imagPart };
	return result;
}

/*+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
*   addComplex()
*   @param: _Dcomplex, _Dcomplex
*   @return: _Dcomplex
*   Description: adds 2 complex numbers
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-*/
_Dcomplex addComplex(_Dcomplex a, _Dcomplex b) {
	double realPart = creal(a) + creal(b);
	double imagPart = cimag(a) + cimag(b);

	_Dcomplex result = { realPart, imagPart };
	return result;
}


/*+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
*   printFFT()
*   @param: _Dcomplex, int
*   @return: none
*   Description: prints the FFT (components of the signal etc..)
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-*/
void printFFT(const _Dcomplex * fft, int N) {
	for (int i = 0; i < N; i++) {
		printf("X(%i) = %f + %fi\n", i, creal(fft[i]), cimag(fft[i]));
	}
}


