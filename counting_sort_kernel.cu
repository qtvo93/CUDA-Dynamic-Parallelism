/* Write GPU code to perform the step(s) involved in counting sort. 
 Add additional kernels and device functions as needed. */

__global__ void counting_sort_kernel(int *histogram, int *input_array, int length)
{
  int threadx = blockIdx.x * blockDim.x + threadIdx.x;
  if (length <= threadx) {
    return;
  }
  atomicAdd(&histogram[input_array[threadx]], 1);
}
__global__ void bins_to_inclusive(int *histogram, int length) {
  int threadx = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadx == 0) {
    for (int i = 0; i < length; i++) {
      histogram[i] = histogram[i - 1] + histogram[i];
    }
  }
}
__global__ void inclusive_to_sorted(int *histogram, int *sorted_array) {
  int threadx = blockIdx.x * blockDim.x + threadIdx.x;
  int start_idx = 0;
  if (threadx != 0) {
    start_idx = histogram[threadx - 1];
  }
  for (int j = start_idx; j < histogram[threadx]; j++) {
    atomicAdd(&sorted_array[j],threadx);
  }
  return;
}
