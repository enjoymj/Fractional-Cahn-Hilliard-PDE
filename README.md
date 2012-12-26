Fractional-Cahn-Hilliard-PDE
============================

pesudo spectral method for Fractional-Cahn-Hilliard-PDE 

It contains basic 1D radix-4 fft and some variants of it in OPENCL targeting GPU devices
Radix-8 C implementation is also included.

For 2D fft, hierarchy FFT is implemented targeting 64\* 64, 256 \*256 and 1024\*1024 matrix.
Radix-4 2D fft is implemented for any matrix size of power of 4

Makefile in this directory with generate executables ch2d and test

test will compare performance of different 2DFFT implementation

ch2d will take in three variables, specific Soblev space, epsilon level and matrix size.
Here matrix size is restricted to 64, 256 and 1024. 

Codes are hard coded towards specific devices, but can be easily changed to compile on other machine


Last modified 12/25/2012 by Kangping Zhu

