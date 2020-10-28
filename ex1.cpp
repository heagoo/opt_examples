/*
 * g++ ex1.cpp -std=c++11 -fopenmp -O2 -mavx512f -o sm_gcc
 * icpc ex1.cpp -std=c++11 -fopenmp -O2 -mavx512f -o sm_icc
 * OMP_NUM_THREADS=8 taskset -c 0-7 ./sm_gcc
 * OMP_NUM_THREADS=8 taskset -c 0-7 ./sm_icc
 */
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <omp.h>
#include "timer.h"

static void init(float *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = 1.0f * rand() / RAND_MAX;
    }
}

// Copied and revised from BERT impl.
// https://github.com/intel/light-model-transformer/tree/master/BERT
void computeSoftmax(float **data, float *exp_buffer, int tokenSize) {
    #pragma omp parallel for
    for (int i = 0; i < 12; ++i) {
        for (int row = 0; row < tokenSize; ++row) {
            int tid = omp_get_thread_num();
            float *pbuffer = &exp_buffer[tid * tokenSize];

            // max_val is used to avoid exp(x) = inf
            float max_val = -std::numeric_limits<float>::max();

            #pragma omp simd
            for (int j = 0; j < tokenSize; ++j) {
                if (data[i][row*tokenSize+j] > max_val) {
                    max_val = data[i][row*tokenSize+j];
                }
            }

            float sum = 0;
            #pragma omp simd
            for (int j = 0; j < tokenSize; ++j) {
                pbuffer[j] = exp(data[i][row*tokenSize+j] - max_val);
                sum += pbuffer[j];
            }

            float r_sum = 1.0f / sum;

            #pragma omp simd
            for (int j = 0; j < tokenSize; ++j) {
                data[i][row*tokenSize+j] = pbuffer[j] * r_sum;
            }
        }
    }
}

int main() {
  int num_threads;
  const int tokenSize = 128;
  
  float *data = (float *)aligned_alloc(64, 12 * tokenSize * tokenSize * sizeof(float));
  init(data, 12 * tokenSize * tokenSize);

  float *pdata[12];
  for (int i = 0; i < 12; ++i) {
    pdata[i] = &data[i * tokenSize * tokenSize];
  }

  // Get thread number
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    if (tid == 0) { num_threads = omp_get_num_threads(); }
  }

  float *exp_buffer = (float *)aligned_alloc(64, num_threads * tokenSize * sizeof(float));

  // Warm up
  for (int i = 0; i < 10; ++i) {
    computeSoftmax(pdata, exp_buffer, tokenSize);
  }

  // Benchmark
  {
    Timer t("Softmax");
    for (int i = 0; i < 100; ++i) {
      computeSoftmax(pdata, exp_buffer, tokenSize);
    }
  }

  free(exp_buffer);
  free(data);
}
