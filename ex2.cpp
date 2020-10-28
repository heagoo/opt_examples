/*
 * g++ -DV1 ex2.cpp -O2 -o gemv_v1
 * g++ -DV2 ex2.cpp -mavx512f -O2 -o gemv_v2
 * g++ -DV3 ex2.cpp -mavx512f -O2 -o gemv_v3
 */
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <immintrin.h>
#include "timer.h"

static void init(float *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = 1.0f * rand() / RAND_MAX;
    }
}

#ifdef V1
// y[m] = w[m][n] * x[n]
void gemv(const float *w, const float *x, float *y, int m, int n) {
  for (int i = 0; i < m; ++i) {
    float r = 0;
    for (int j = 0; j < n; ++j) {
      //r += w[i][j] * x[j];
      r += w[i * n + j] * x[j];
    }
    y[i] = r;
  }
}
#elif V2
inline float horizontal_add(__m512 a) { 
  __m512 tmp = _mm512_add_ps(a, _mm512_shuffle_f32x4(a, a, _MM_SHUFFLE(0, 0, 3, 2)));
  __m128 r = _mm512_castps512_ps128(_mm512_add_ps(tmp, _mm512_shuffle_f32x4(tmp, tmp, _MM_SHUFFLE(0, 0, 0, 1))));
  r = _mm_hadd_ps(r, r);
  return _mm_cvtss_f32(_mm_hadd_ps(r, r));
}

void gemv(const float *w, const float *x, float *y, int m, int n) {
  assert(n % 16 == 0);
  for (int i = 0; i < m; ++i) {
    __m512 vy = _mm512_set1_ps(0);
    for (int j = 0; j < n; j += 16) {
      __m512 vw = _mm512_loadu_ps(&w[i * n + j]);
      __m512 vx = _mm512_loadu_ps(&x[j]);
      vy = _mm512_fmadd_ps(vw, vx, vy);
    }
    y[i] = horizontal_add(vy);
  }
}
#elif V3
static void transpose_simple(float *w, int m, int n) {
  assert(m == n);
  for (int i = 0; i < m; ++i) {
    for (int j = i + 1; j < n; ++j) {
      // Swap w[i][j] and w[j][i]
      float t = w[i * n + j];
      w[i * n + j] = w[j * n + i];
      w[j * n + i] = t;
    }
  }
}

// As w is transposed, so its shape is n*m (n rows, m cols)
void gemv(const float *w, const float *x, float *y, int m, int n) {
  assert(m % 16 == 0);
  assert(m <= 24 * 16); // For current simple impl.

  int blocks = m / 16;
  __m512 vy[24];
  for (int i = 0; i < blocks; ++i) {
    vy[i] = _mm512_set1_ps(0);
  }

  for (int i = 0; i < n; ++i) {
    __m512 vx = _mm512_set1_ps(x[i]);
    for (int j = 0; j < blocks; ++j) {
      __m512 vw = _mm512_loadu_ps(&w[i * m + j * 16]);
      vy[j] = _mm512_fmadd_ps(vw, vx, vy[j]);
    }
  }

  for (int i = 0; i < blocks; ++i) {
    _mm512_storeu_ps(&y[i * 16], vy[i]);
  }
}
#endif

int main() {
  const int m = 256;
  const int n = 256;
  float *w = new float[m * n];
  float *x = new float[n];
  float *y = new float[m];

  init(w, m * n);
  init(w, n);

#ifdef V3
  transpose_simple(w, m, n);
#endif

  // Warm up
  for (int i = 0; i < 10; ++i) {
    gemv(w, x, y, m, n);
  }

  // Benchmark
  {
    Timer t("GEMV");
    for (int i = 0; i < 10; ++i) {
      gemv(w, x, y, m, n);
    }
  }

  return 0;
}
