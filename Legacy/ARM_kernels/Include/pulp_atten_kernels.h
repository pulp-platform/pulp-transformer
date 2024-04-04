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

#include "arm_nnsupportfunctions.h"
#include "arm_nn_tables.h"

pulp_atten_linear_2x2_input_reuse(
  const q7_t * pInBuffer,
  const q7_t *  pWeight,
  q7_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  projections,
  const uint16_t  heads
);
pulp_atten_linear_4x2_input_reuse(
  const q7_t * pInBuffer,
  const q7_t *  pWeight,
  const q15_t * pInBuffer_15,
  const q15_t *  pWeight_15,
  q7_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  projections,
  const uint16_t  heads
);
pulp_atten_linear_4x2_weights_reuse(
  const q7_t * pInBuffer,
  const q7_t *  pWeight,
  const q15_t * pInBuffer_15,
  const q15_t *  pWeight_15,
  q7_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  projections,
  const uint16_t  heads
);
pulp_atten_matmul_4x2_sequential(
  const q7_t * pInBuffer,
  const q7_t *  pWeight,
  const q15_t * pInBuffer_15,
  const q15_t *  pWeight_15,
  q7_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  projections,
  const uint16_t  heads
);
pulp_atten_matmul_4x2_strided(
  const q7_t * pInBuffer,
  const q7_t *  pWeight,
  const q15_t * pInBuffer_15,
  const q15_t *  pWeight_15,
  q7_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  projections,
  const uint16_t  heads
);
pulp_atten_linear_4x1_input_reuse(
  const q7_t * pInBuffer,
  const q7_t *  pWeight,
  const q15_t * pInBuffer_15,
  const q15_t *  pWeight_15,
  q7_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  projections,
  const uint16_t  heads
);
pulp_atten_linear_4x1_weights_reuse(
  const q7_t * pInBuffer,
  const q7_t *  pWeight,
  const q15_t * pInBuffer_15,
  const q15_t *  pWeight_15,
  q7_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  projections,
  const uint16_t  heads
);
pulp_atten_matmul_4x1_sequential(
  const q7_t * pInBuffer,
  const q7_t *  pWeight,
  const q15_t * pInBuffer_15,
  const q15_t *  pWeight_15,
  q7_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  projections,
  const uint16_t  heads
);
pulp_atten_matmul_4x1_strided(
  const q7_t * pInBuffer,
  const q7_t *  pWeight,
  const q15_t * pInBuffer_15,
  const q15_t *  pWeight_15,
  q7_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  projections,
  const uint16_t  heads
);

void  IntSoftmax(
  int32_t * __restrict__ pInBuffer,
  const int32_t dimension,
  int32_t * __restrict__ pOutBuffer,
  int32_t * __restrict__ coeffs );