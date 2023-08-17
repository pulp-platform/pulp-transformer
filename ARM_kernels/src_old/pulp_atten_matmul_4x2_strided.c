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

#include "arm_math.h"
#include "arm_nnfunctions.h"

pulp_atten_matmul_4x2_strided(
  const q7_t * pInBuffer,
  const q7_t *  pWeight,
  q7_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  projections,
  const uint16_t  heads
) 
{

  // local vars
  int seq_out, proj_out, seq_out_internal, head_out;
  int8_t *pA, *pA2;
  int8_t *pB, *pB2, *pB3, *pB4;
  int8_t *pOut = pOutBuffer;
  int8_t *pOut2 = pOut + dim_sequence * heads;
  
  for (head_out = 0; head_out < heads; head_out++)
  {  
    for (seq_out = 0; seq_out < (dim_sequence>>1); seq_out++)
    {
      pB = pWeight + (head_out * dim_sequence * projections);
      pOut = pOutBuffer + head_out * dim_sequence + seq_out * heads * dim_sequence * 2;
      pOut2 = pOut + heads * dim_sequence;
      for (seq_out_internal = 0; seq_out_internal < (dim_sequence>>2); seq_out_internal++)
      {  
        int sum = 0;
        int sum2 = 0;
        int sum3 = 0;
        int sum4 = 0;
        int sum5 = 0;
        int sum6 = 0;
        int sum7 = 0;
        int sum8 = 0;
        pB2 = pB + projections;
        pB3 = pB2 + projections;
        pB4 = pB3 + projections;
        pA = pInBuffer + head_out * dim_sequence * projections + 2 * seq_out * projections;
        pA2 = pA + projections;
        for (proj_out = 0; proj_out < (projections>>2); proj_out++)
        {
            q31_t     inA11, inA12, inA21, inA22;
            q31_t     inB11, inB12, inB21, inB22, inB31, inB32, inB41, inB42;
            pB =  (q7_t *) read_and_pad((void *)pB,  &inB11, &inB12);
            pB2 = (q7_t *) read_and_pad((void *)pB2, &inB21, &inB22);
            pB3 = (q7_t *) read_and_pad((void *)pB3, &inB31, &inB32);
            pB4 = (q7_t *) read_and_pad((void *)pB4, &inB41, &inB42);

            pA =  (q7_t *) read_and_pad((void *)pA,  &inA11, &inA12);
            pA2 = (q7_t *) read_and_pad((void *)pA2, &inA21, &inA22);

            sum  = __SMLAD(inA11, inB11, sum );
            sum2 = __SMLAD(inA11, inB21, sum2);
            sum3 = __SMLAD(inA11, inB31, sum3);
            sum4 = __SMLAD(inA11, inB41, sum4);
            sum5 = __SMLAD(inA21, inB11, sum5);
            sum6 = __SMLAD(inA21, inB21, sum6);
            sum7 = __SMLAD(inA21, inB31, sum7);
            sum8 = __SMLAD(inA21, inB41, sum8);

            sum  = __SMLAD(inA12, inB12, sum );
            sum2 = __SMLAD(inA12, inB22, sum2);
            sum3 = __SMLAD(inA12, inB32, sum3);
            sum4 = __SMLAD(inA12, inB42, sum4);
            sum5 = __SMLAD(inA22, inB12, sum5);
            sum6 = __SMLAD(inA22, inB22, sum6);
            sum7 = __SMLAD(inA22, inB32, sum7);
            sum8 = __SMLAD(inA22, inB42, sum8);
        }
        *pOut++ = (q7_t) __SSAT(sum ,8);
        *pOut++ = (q7_t) __SSAT(sum2,8);
        *pOut++ = (q7_t) __SSAT(sum3,8);
        *pOut++ = (q7_t) __SSAT(sum4,8);
        *pOut2++ = (q7_t)__SSAT(sum5,8);
        *pOut2++ = (q7_t)__SSAT(sum6,8);
        *pOut2++ = (q7_t)__SSAT(sum7,8);
        *pOut2++ = (q7_t)__SSAT(sum8,8);
        pB = pB + (3 * projections);
      }
    }
  }
}
