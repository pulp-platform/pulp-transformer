/*
 * pulp_atten_kernel.h
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
#include "pulp.h"

void __attribute__ ((noinline)) pulp_atten_linear_2x1_input_reuse(
  const int8_t * pInBuffer,
  const int8_t *  pWeight,
  int8_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  projections,
  const uint16_t  heads
);

void __attribute__ ((noinline)) pulp_atten_linear_2x1_weights_reuse(
  const int8_t * pInBuffer,
  const int8_t *  pWeight,
  int8_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  projections,
  const uint16_t  heads
);

void __attribute__ ((noinline)) pulp_atten_matmul_2x1_sequential(
  const int8_t * pInBuffer,
  const int8_t *  pWeight,
  int8_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  projections,
  const uint16_t  heads
);

void __attribute__ ((noinline)) pulp_atten_matmul_2x1_strided(
  const int8_t * pInBuffer,
  const int8_t *  pWeight,
  int8_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  projections,
  const uint16_t  heads
);

void __attribute__ ((noinline)) pulp_atten_linear_4x2_input_reuse(
  const int8_t * pInBuffer,
  const int8_t *  pWeight,
  int8_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  projections,
  const uint16_t  heads
);
void __attribute__ ((noinline)) pulp_atten_linear_4x2_out(
  const int8_t * pInBuffer,
  const int8_t *  pWeight,
  int8_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  projections,
  const uint16_t  heads
);

void __attribute__ ((noinline)) pulp_atten_linear_4x2_weights_reuse(
  const int8_t * pInBuffer,
  const int8_t *  pWeight,
  int8_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  projections,
  const uint16_t  heads
);

void __attribute__ ((noinline)) pulp_atten_matmul_4x2_sequential(
  const int8_t * pInBuffer,
  const int8_t *  pWeight,
  int8_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  projections,
  const uint16_t  heads
);

void __attribute__ ((noinline)) pulp_atten_matmul_4x2_strided(
  const int8_t * pInBuffer,
  const int8_t *  pWeight,
  int8_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  projections,
  const uint16_t  heads
);

void __attribute__ ((noinline))  IntSoftmax(
  int32_t * pInBuffer,
  const uint16_t dimension,
  int32_t * pOutBuffer,
  int32_t * coeffs );