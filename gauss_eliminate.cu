/*
 *  Host-side code for Gaussian elimination. 
 * 
 * Author: Naga Kandasamy
 * Date modified: March 2, 2021
 * 
 * Student name(s): Quoc Thinh Vo
 * Date modified: 03/15/2021
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include <device_launch_parameters.h>


#include "gauss_eliminate_kernel.cu"

#define MIN_NUMBER 2
#define MAX_NUMBER 50
#define THREADS_PER_BLOCK 16


extern "C" int compute_gold(float*, const float*, unsigned int);
Matrix allocate_matrix_on_gpu(const Matrix M);
Matrix allocate_matrix(int num_rows, int num_columns, int init);
void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost);
void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice);
void gauss_eliminate_on_device(const Matrix M, Matrix P);
int perform_simple_check(const Matrix M);
void print_matrix(const Matrix M);
void write_matrix_to_file(const Matrix M);
float get_random_number(int, int);
void check_CUDA_error(const char *msg);
int check_results(float *reference, float *gpu_result, int num_elements, float threshold);


int main(int argc, char** argv) 
{
    if (argc > 1) {
        printf("Error. This program accepts no arguments.\n");
        exit(EXIT_SUCCESS);
    }
	
    Matrix  A; /* The N x N input matrix */
	Matrix  U; /* The upper triangular matrix returned by device */ 
	
	/* Allocate and initialize the matrices */
    srand(time(NULL));
	struct timeval start, stop;
	gettimeofday(&start, NULL);
	A  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 1);
	U  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); 

	/* Perform Gaussian elimination on the CPU */
	Matrix reference = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);
	int status = compute_gold(reference.elements, A.elements, A.num_rows);
	if (status == 0) { 
		printf("Failed to convert given matrix to upper triangular. Try again. Exiting. \n");
		exit(EXIT_FAILURE);
	}
	
    status = perform_simple_check(reference); // Check that the principal diagonal elements are 1 
	if (status == 0) {
		printf("The upper triangular matrix is incorrect. Exiting. \n");
		exit(EXIT_FAILURE); 
	}
	
    printf("Gaussian elimination on the CPU was successful. \n");
	gettimeofday(&stop, NULL);
	printf("CPU time = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));

	/* Perform Gaussin elimination on device. Return the result in U. */
	
	gettimeofday(&start, NULL);
	gauss_eliminate_on_device(A, U);
	printf("Gaussian elimination on the GPU was successful. \n");
	gettimeofday(&stop, NULL);
	printf("GPU time = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));
    
	/* Check if device result matches reference. */
	int num_elements = MATRIX_SIZE*MATRIX_SIZE;
    int res = check_results(reference.elements, U.elements, num_elements, 0.001f);
    printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

	/* Free host matrices. */
	free(A.elements); 
	free(U.elements); 
	free(reference.elements);

    exit(EXIT_SUCCESS);
}

/* FIXME: complete this function. */
void gauss_eliminate_on_device(const Matrix A, Matrix U)
{
	Matrix UD = allocate_matrix(U.num_rows,U.num_columns,0);
	copy_matrix_to_device(UD, U);
	
	int  k;
	
	dim3 threads(TILE_SIZE, TILE_SIZE);
	dim3 grid((MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE, (MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE);
	
	float *dI; 
	const int n = MATRIX_SIZE;
	for (k = 0; k<n; k++){
		division__kernel << < grid, threads >>>(UD.elements, dI, n, k);
		elimination_kernel << < grid, threads >>>(UD.elements, dI, n, k);
		check_CUDA_error("error found");
		cudaDeviceSynchronize();
	}
	

	copy_matrix_from_device(U, UD);
	
}

Matrix allocate_matrix_on_gpu(const Matrix M){
	Matrix Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}	

/* Allocate matrix of dimensions height * width
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
Matrix allocate_matrix(int num_rows, int num_columns, int init)
{
    Matrix M;
    M.num_columns = M.pitch = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
		
	M.elements = (float*)malloc(size*sizeof(float));
	for (unsigned int i = 0; i < size; i++) {
		if (init == 0) 
            M.elements[i] = 0; 
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
    
    return M;
}	

/* Copy matrix to from host to device */
void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(float);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

/* Copy matrix from device to host */
void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

/* Print matrix to screen */
void print_matrix(const Matrix M)
{
	for (unsigned int i = 0; i < M.num_rows; i++){
		for (unsigned int j = 0; j < M.num_columns; j++)
			printf("%f ", M.elements[i*M.num_rows + j]);
		printf("\n");
	} 
	printf("\n");
}

/* Return a random number between [min, max] */ 
float get_random_number(int min, int max)
{
	return (float)floor((double)(min + (max - min + 1)*((float)rand()/(float)RAND_MAX)));
}

/* Check to see if the principal diagonal elements are 1 */
int perform_simple_check(const Matrix M)
{
	for (unsigned int i = 0; i < M.num_rows; i++)
        if ((fabs(M.elements[M.num_rows*i + i] - 1.0)) > 0.001) return 0;
	
    return 1;
} 

void check_CUDA_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) 
	{
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}						 
}

int check_results(float *reference, float *gpu_result, int num_elements, float threshold)

{

    int i;

    int check = 1;

    float epsilon = 0.0;

   

    for (i = 0; i < num_elements; i++)

        if (fabsf(reference[i] - gpu_result[i]) > threshold) {

           check = 0;

            break;

       }

   for (i = 0; i < num_elements; i++)

       if (fabsf(reference[i] - gpu_result[i]) > epsilon) {

           epsilon = fabsf(reference[i] - gpu_result[i]);

   }

   printf("Max epsilon = %f. \n", epsilon);

   return check;

}
