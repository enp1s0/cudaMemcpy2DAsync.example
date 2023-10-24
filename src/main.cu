#include <iostream>
#include <stdexcept>
#include <sstream>
#include <cuda_device_runtime_api.h>
#include <cuda.h>

#define CUDA_CHECK_ERROR(status) cuda_error_check(status, __FILE__, __LINE__, __func__)

inline void cuda_error_check(cudaError_t error, const std::string filename, const std::size_t line, const std::string funcname, const std::string message = ""){
  if(error != cudaSuccess){
    std::stringstream ss;
    ss<< cudaGetErrorString( error );
    ss<<" ["<<filename<<":"<<line<<" in "<<funcname<<"]";
    throw std::runtime_error(ss.str());
  }
}

constexpr std::size_t A_M = 1lu << 5;
constexpr std::size_t A_N = 1lu << 4;

constexpr std::size_t B_M = 1lu << 2;
constexpr std::size_t B_N = 1lu << 2;

constexpr std::size_t B_OFFSET_M = 1lu << 2;
constexpr std::size_t B_OFFSET_N = 1lu << 2;

// Copy a part of matrix A as matrix B

// Col-major
//                     A_N
//     +--------------------------------------+
//     | A                                    |
//     |                                      |
//     | (B_OFFSET_M, B_OFFSET_N)             |
//     |           +------------------+       |
//     |           | B                |       |
//     |           |                  |       |
//     |           |                  |       |
// A_M |       B_M |                  |       |
//     |           |                  |       |
//     |           |                  |       |
//     |           +------------------+       |
//     |                   B_N                |
//     |                                      |
//     |                                      |
//     |                                      |
//     +--------------------------------------+

using data_t = float;

void print_matrix(
    const data_t* mat_ptr,
    const std::size_t ld,
    const std::size_t m,
    const std::size_t n
    ) {

  for (std::size_t i = 0; i < m; i++) {
    for (std::size_t j = 0; j < n; j++) {
      std::printf("%+.3e ", mat_ptr[i + j * ld]);
    }
    std::printf("\n");
  }
}

int main() {
  data_t* mat_A_ptr, *mat_B_ptr;
  CUDA_CHECK_ERROR(cudaMallocManaged(&mat_A_ptr, sizeof(data_t) * A_M * A_N));
  CUDA_CHECK_ERROR(cudaMallocManaged(&mat_B_ptr, sizeof(data_t) * B_M * B_N));

  for (std::size_t i = 0; i < A_M * A_N; i++) {
    mat_A_ptr[i] = i;
  }

  // Note: Consider that the matrices are stored in row-major
  CUDA_CHECK_ERROR(cudaMemcpy2D(
      mat_B_ptr, sizeof(data_t) * B_M,
      mat_A_ptr + B_OFFSET_N * A_M + B_OFFSET_M, sizeof(data_t) * A_M,
      B_M * sizeof(data_t),
      B_N,
      cudaMemcpyDefault
      ));

  std::printf("A_M = %lu\n", A_M);
  std::printf("A_N = %lu\n", A_N);
  std::printf("B_M = %lu\n", B_M);
  std::printf("B_N = %lu\n", B_N);
  std::printf("B_OFFSET_M = %lu\n", B_OFFSET_M);
  std::printf("B_OFFSET_N = %lu\n", B_OFFSET_M);

  std::printf("A\n");
  print_matrix(mat_A_ptr, A_M, A_M, A_N);
  std::printf("B\n");
  print_matrix(mat_B_ptr, B_M, B_M, B_N);
  std::printf("Correct Upper Left = %e\n", static_cast<double>(B_OFFSET_N * A_M + B_OFFSET_M));


  CUDA_CHECK_ERROR(cudaFree(mat_A_ptr));
  CUDA_CHECK_ERROR(cudaFree(mat_B_ptr));
}
