 /* Device code. */
#include "gauss_eliminate.h"

__global__ void division__kernel(float *U, float* current_row, int k, int offset)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < k && y < k)
	if (x == y && x == offset){
		current_row[x*k + y] /= U[offset*k + offset];
		U[x*k + y] /= U[offset*k + offset];
	}
}


__global__ void elimination_kernel(float *U, float *current_row, int k, int offset)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < k && y < k){
		if (x != offset){
			current_row[x*k + y] -= current_row[offset*k + y] * U[x*k + offset];
			if (y != offset){
				U[x*k + y] -= U[offset*k + y] * U[x*k + offset];
			}	 
		}
	