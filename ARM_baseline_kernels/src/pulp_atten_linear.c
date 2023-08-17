/*
 * pulp_atten_linear.c
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

#include "arm_math.h"
#include "arm_nnfunctions.h"

pulp_atten_linear(
  const q7_t * pInBuffer,
  const q7_t *  pWeight,
  q7_t *       pOutBuffer_support,
  q7_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  projections,
  const uint16_t  heads
) 
{

  // local vars
  int seq_out;
  q15_t pOut_linear_15[projections*heads];
  for (seq_out = 0; seq_out < dim_sequence; seq_out++)
  {  
    q7_t *pIn_linear = pInBuffer + seq_out*dim_embedding;
    q7_t *pOut_linear = pOutBuffer_support + seq_out*projections*heads;
    transformer_fully_connected_q7_opt(pIn_linear,
                               pWeight,
                               dim_embedding,
                               projections*heads,
                               pOut_linear, 
                               pOut_linear_15);
  }
  pulp_atten_reshape(pOutBuffer_support, pOutBuffer, dim_sequence, projections, heads);
}


