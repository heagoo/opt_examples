/*
 * Code is from: https://stackoverflow.com/questions/7327994/prefetching-examples
 * g++ ex4.cpp -DDO_PREFETCH -o with-prefetch -O2
 * g++ ex4.cpp -o no-prefetch -O2
 * perf stat -e L1-dcache-load-misses,L1-dcache-loads ./with-prefetch 
 * perf stat -e L1-dcache-load-misses,L1-dcache-loads ./no-prefetch 
 */
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

int binarySearch(int *array, int number_of_elements, int key) {
  int low = 0, high = number_of_elements - 1, mid;

  while (low <= high) {
    mid = (low + high) / 2;

    #ifdef DO_PREFETCH
    // low path
    __builtin_prefetch (&array[(mid + 1 + high) / 2], 0, 1);
    // high path
    __builtin_prefetch (&array[(low + mid - 1) / 2], 0, 1);
    #endif

    if (array[mid] < key)
      low = mid + 1; 
    else if (array[mid] == key)
      return mid;
    else if (array[mid] > key)
      high = mid - 1;
  }

  return -1;
}

int main() {
  int SIZE = 1024 * 1024 * 512;
  int *array = (int *)malloc(SIZE * sizeof(int));
  for (int i = 0; i < SIZE; i++) {
    array[i] = i;
  }

  int NUM_LOOKUPS = 1024 * 1024 * 8;
  int *lookups = (int *)malloc(NUM_LOOKUPS * sizeof(int));
  //srand(time(NULL));
  srand(1000);
  for (int i = 0; i < NUM_LOOKUPS; i++){
    lookups[i] = rand() % SIZE;
  }

  {
    Timer t("Binary Search");
    for (int i = 0; i < NUM_LOOKUPS; i++){
      int result = binarySearch(array, SIZE, lookups[i]);
    }
  }

  free(array);
  free(lookups);
}
