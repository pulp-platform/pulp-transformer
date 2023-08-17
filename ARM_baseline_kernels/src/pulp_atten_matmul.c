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
#include "pulp_atten_kernels.h"

pulp_atten_matmul(
  const q7_t * pInBuffer,
  const q7_t * pInBuffer2,
  q7_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  projections,
  const uint16_t  heads
) 
{

  // local vars
  int seq_out, head_out;
  q15_t pOut_linear_15[dim_sequence*2];
  for (head_out = 0; head_out < heads; head_out++)
  {
    for (seq_out = 0; seq_out < dim_sequence; seq_out++)
    {  
      q7_t *pIn_linear = pInBuffer + seq_out*projections + head_out*dim_sequence*projections;
      q7_t *pOut_linear = pOutBuffer + seq_out*dim_sequence + head_out*dim_sequence*dim_sequence;
      q7_t *pIn2_linear = pInBuffer2 + head_out*dim_sequence*projections;
    transformer_fully_connected_q7_opt(pIn_linear,
                               pIn2_linear,
                               projections,
                               dim_sequence,
                               pOut_linear, 
                               pOut_linear_15);
      IntSoftmax(pOutBuffer, dim_sequence, pOutBuffer, pInBuffer);
    }
  }
}
