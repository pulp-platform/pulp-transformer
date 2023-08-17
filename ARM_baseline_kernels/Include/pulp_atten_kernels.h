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

pulp_atten_reshape_matmul(
      q7_t *pInBuffer,
      q7_t *pOutBuffer,
      const uint16_t  dim_sequence,
      const uint16_t  projections,
      const uint16_t  heads
);

pulp_atten_reshape_hsp(
      q7_t *pInBuffer,
      q7_t *pOutBuffer,
      const uint16_t  dim_sequence,
      const uint16_t  projections,
      const uint16_t  heads
);

pulp_atten_reshape(
      q7_t *pInBuffer,
      q7_t *pOutBuffer,
      const uint16_t  dim_sequence,
      const uint16_t  projections,
      const uint16_t  heads
);

pulp_atten_matmul_reshape(
  const q7_t * pInBuffer,
  const q7_t * pInBuffer2,
  q7_t *       pOutBuffer_support,
  q7_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  projections,
  const uint16_t  heads
);

pulp_atten_matmul(
  const q7_t * pInBuffer,
  const q7_t * pInBuffer2,
  q7_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  projections,
  const uint16_t  heads
);

pulp_atten_linear_hsp(
  const q7_t * pInBuffer,
  const q7_t *  pWeight,
  q7_t *       pOutBuffer_support,
  q7_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  projections,
  const uint16_t  heads
);

pulp_atten_linear(
  const q7_t * pInBuffer,
  const q7_t *  pWeight,
  q7_t *       pOutBuffer_support,
  q7_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  projections,
  const uint16_t  heads
);

transformer_fully_connected_q7_opt(const q7_t * pV,
                           const q7_t * pM,
                           const uint16_t dim_vec,
                           const uint16_t num_of_rows,
                           q7_t * pOut, 
                           q15_t * vec_buffer);