/*
 * pulp_atten_matmul_4x2_strided.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 *
 * Copyright (C) 2018-2021 University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//#include "pmsis.h"
#include <stdint.h>


int __attribute__ ((noinline))  int_exp(
  int32_t * __restrict__ pInBuffer,
  const int32_t dimension,
  const int32_t * __restrict__ coeff,
  const int32_t x0)
{
  int32_t q;
  int32_t r;
  int32_t n = 10;
  int32_t factor = (1<<16)/x0; // avoid division
  int sum = 0;
  for (int i = 0; i < dimension; i++)
  {
    int32_t tmp = *pInBuffer;
    q = (tmp*factor)>>16; //q[i] = (*(pInBuffer+i)/x0);
    r = tmp - x0*q;
    r = (r + coeff[0]) * r + coeff[1];
    tmp = r << (n-q);
    *pInBuffer++=tmp;
    sum += tmp;
  }
  return sum;
}

int max_vector(
  const int32_t * __restrict__ pInBuffer,
  const int32_t dimension)
{
  int32_t current_max = 0;
  for (int i = 0; i < dimension; i++) {
    int32_t tmp = *pInBuffer++;
    if (tmp > current_max)
        current_max = tmp;
  }
  return current_max;
}

void __attribute__ ((noinline))  IntSoftmax(
  int32_t * __restrict__ pInBuffer,
  const int32_t dimension,
  int32_t * __restrict__ pOutBuffer,
  int32_t * __restrict__ coeffs )
{
  int32_t x0 = -24; // floor(-0.6931 / 0.03) scaling factor #ln2
  int32_t x_int_max;
  int32_t n = 10;
  int32_t low_limit = x0*n;

  x_int_max = max_vector(pInBuffer, dimension);
  int32_t *pInBuffer_base = pInBuffer;
  for (int i = 0; i < dimension; i++)
  {
    int32_t tmp = *pInBuffer;
    tmp = tmp-x_int_max;
    if (tmp < low_limit) // maybe can use some clipping function + sum with low_limit
        tmp = low_limit;
    *pInBuffer++ = tmp;
  }
  pInBuffer = pInBuffer_base;
  int sum = int_exp(pInBuffer, dimension, coeffs, x0);
  int32_t factor = (1<<30)/sum; 
  for (int i = 0; i < dimension; i++)
  {
    int32_t tmp = *pInBuffer++;
    *pOutBuffer++ = (tmp * factor) >> 22;
  }
}