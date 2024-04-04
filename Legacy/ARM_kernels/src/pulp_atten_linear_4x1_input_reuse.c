/*
 * pulp_atten_linear_4x2_input_reuse.c
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
) 
{

  // local vars
  int head_out, proj_out, seq_out, emb;
  int8_t *pA, *pA2;
  int8_t *pB, *pB2, *pB3, *pB4;
  int8_t *pOut = pOutBuffer;
  int8_t *pOut2 = pOut + projections;

  arm_q7_to_q15_reordered_no_shift(pInBuffer, pInBuffer_15, dim_sequence*dim_embedding);
  arm_q7_to_q15_reordered_no_shift(pWeight, pWeight_15, dim_embedding*heads*projections);
  for (head_out = 0; head_out < heads; head_out++)
  {
    for (seq_out = 0; seq_out < (dim_sequence); seq_out++)
    {  
      pB = pWeight_15 + (head_out * dim_embedding * projections);
      pOut = pOutBuffer + head_out * projections * dim_sequence + seq_out * projections;
      pOut2 = pOut + projections;
      for (proj_out = 0; proj_out < (projections>>2); proj_out++)
      {  
        int sum = 0;
        int sum2 = 0;
        int sum3 = 0;
        int sum4 = 0;
        int sum5 = 0;
        int sum6 = 0;
        int sum7 = 0;
        int sum8 = 0;
        pB2 = pB + dim_embedding;
        pB3 = pB2 + dim_embedding;
        pB4 = pB3 + dim_embedding;
        pA = pInBuffer_15 + seq_out * dim_embedding;
        for (emb = 0; emb < (dim_embedding>>2); emb++)
        { 
            q31_t     inA11, inA12;
            q31_t     inB11, inB12, inB21, inB22, inB31, inB32, inB41, inB42;
            inA11 = *__SIMD32(pA)++;
            inA12 = *__SIMD32(pA)++;
            inB11 = *__SIMD32(pB)++;
            inB12 = *__SIMD32(pB)++;
            inB21 = *__SIMD32(pB2)++;
            inB22 = *__SIMD32(pB2)++;
            inB31 = *__SIMD32(pB3)++;
            inB32 = *__SIMD32(pB3)++;
            inB41 = *__SIMD32(pB4)++;
            inB42 = *__SIMD32(pB4)++;

            sum  = __SMLAD(inA11, inB11, sum );
            sum2 = __SMLAD(inA11, inB21, sum2);
            sum3 = __SMLAD(inA11, inB31, sum3);
            sum4 = __SMLAD(inA11, inB41, sum4);

            sum  = __SMLAD(inA12, inB12, sum );
            sum2 = __SMLAD(inA12, inB22, sum2);
            sum3 = __SMLAD(inA12, inB32, sum3);
            sum4 = __SMLAD(inA12, inB42, sum4);
        }
        *pOut++ = (q7_t) __SSAT(sum ,8);
        *pOut++ = (q7_t) __SSAT(sum2,8);
        *pOut++ = (q7_t) __SSAT(sum3,8);
        *pOut++ = (q7_t) __SSAT(sum4,8);
        pB = pB + (3 * dim_embedding);
      }
    }
  }
}
