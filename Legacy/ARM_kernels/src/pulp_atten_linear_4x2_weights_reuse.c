/*
 * pulp_atten_linear_4x2_weights_reuse.c
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
) 
{

  // local vars
  int head_out, proj_out, seq_out, emb;
  int8_t *pA, *pA2, *pA3, *pA4;
  int8_t *pB, *pB2;
  int8_t *pOut = pOutBuffer;
  int8_t *pOut2 = pOut + dim_sequence;
  arm_q7_to_q15_reordered_no_shift(pInBuffer, pInBuffer_15, dim_sequence*dim_embedding);
  arm_q7_to_q15_reordered_no_shift(pWeight, pWeight_15, dim_embedding*heads*projections);

  for (head_out = 0; head_out < heads; head_out++)
  {
    for (proj_out = 0; proj_out < (projections>>1); proj_out++)
    {  
      pA = pInBuffer_15;
      pOut = pOutBuffer + head_out * projections * dim_sequence + 2 * proj_out * dim_sequence;
      pOut2 = pOut + dim_sequence;
      for (seq_out = 0; seq_out < (dim_sequence>>2); seq_out++)
      {  
        int sum = 0;
        int sum2 = 0;
        int sum3 = 0;
        int sum4 = 0;
        int sum5 = 0;
        int sum6 = 0;
        int sum7 = 0;
        int sum8 = 0;
        pA2 = pA + dim_embedding;
        pA3 = pA2 + dim_embedding;
        pA4 = pA3 + dim_embedding;
        pB = pWeight_15 + (head_out * dim_embedding * projections) + (dim_embedding * proj_out * 2);
        pB2 = pB + dim_embedding;
        for (emb = 0; emb < (dim_embedding>>2); emb++)
        { 
          q31_t     inA11, inA12, inA21, inA22, inA31, inA32, inA41, inA42;
          q31_t     inB11, inB12, inB21, inB22;
          inA11 = *__SIMD32(pA)++;
          inA12 = *__SIMD32(pA)++;
          inA21 = *__SIMD32(pA2)++;
          inA22 = *__SIMD32(pA2)++;
          inA31 = *__SIMD32(pA3)++;
          inA32 = *__SIMD32(pA3)++;
          inA41 = *__SIMD32(pA4)++;
          inA42 = *__SIMD32(pA4)++;
          inB11 = *__SIMD32(pB)++;
          inB12 = *__SIMD32(pB)++;
          inB21 = *__SIMD32(pB2)++;
          inB22 = *__SIMD32(pB2)++;

          sum  = __SMLAD(inA11, inB11, sum );
          sum2 = __SMLAD(inA21, inB11, sum2);
          sum3 = __SMLAD(inA31, inB11, sum3);
          sum4 = __SMLAD(inA41, inB11, sum4);
          sum5 = __SMLAD(inA11, inB21, sum5);
          sum6 = __SMLAD(inA21, inB21, sum6);
          sum7 = __SMLAD(inA31, inB21, sum7);
          sum8 = __SMLAD(inA41, inB21, sum8);

          sum  = __SMLAD(inA12, inB12, sum );
          sum2 = __SMLAD(inA22, inB12, sum2);
          sum3 = __SMLAD(inA32, inB12, sum3);
          sum4 = __SMLAD(inA42, inB12, sum4);
          sum5 = __SMLAD(inA12, inB22, sum5);
          sum6 = __SMLAD(inA22, inB22, sum6);
          sum7 = __SMLAD(inA32, inB22, sum7);
          sum8 = __SMLAD(inA42, inB22, sum8);
        }
        *pOut++ = (q7_t) __SSAT(sum ,8);
        *pOut++ = (q7_t) __SSAT(sum2,8);
        *pOut++ = (q7_t) __SSAT(sum3,8);
        *pOut++ = (q7_t) __SSAT(sum4,8);
        *pOut2++ = (q7_t)__SSAT(sum5,8);
        *pOut2++ = (q7_t)__SSAT(sum6,8);
        *pOut2++ = (q7_t)__SSAT(sum7,8);
        *pOut2++ = (q7_t)__SSAT(sum8,8);
        pA = pA + (3 * dim_embedding);
      }
    }
  }
}
