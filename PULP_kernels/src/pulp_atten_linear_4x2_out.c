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

#include "pmsis.h"

#define min(a,b) ((a)<(b)?(a):(b))
#define SumDotp(a, b, c) __builtin_pulp_sdotsp4(a, b, c)
#define clip8(x) __builtin_pulp_clip_r(x, 127)

void __attribute__ ((noinline)) pulp_atten_linear_4x2_out(
  const int8_t * pInBuffer,
  const int8_t *  pWeight,
  int8_t *       pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  projections,
  const uint16_t  heads
) 
{
  int core_id = pi_core_id();
  int Log2Core = log2(NUM_CORES);
  int embs_per_core = (dim_embedding >> Log2Core) + ((dim_embedding & (NUM_CORES-1))!=0);
  int start_emb, stop_emb;
  start_emb = min(embs_per_core * core_id, dim_embedding);
  stop_emb = min(start_emb + embs_per_core, dim_embedding);

  // local vars
  int proj_head_in, seq_out, emb_out;
  int8_t *pA, *pA2;
  int8_t *pB, *pB2, *pB3, *pB4;
  int8_t *pOut = pOutBuffer;
  int8_t *pOut2 = pOut + dim_embedding;
  v4s vecA, vecA2;
  v4s vecB, vecB2, vecB3, vecB4;

  for (seq_out = 0; seq_out < (dim_sequence>>1); seq_out++)
  {  
    pOut = pOutBuffer + seq_out * dim_embedding;
    pOut2 = pOut + dim_embedding;
    pB = pWeight;
    for (emb_out = start_emb; emb_out < (stop_emb>>2); emb_out++)
    {
      int sum = 0;
      int sum2 = 0;
      int sum3 = 0;
      int sum4 = 0;
      int sum5 = 0;
      int sum6 = 0;
      int sum7 = 0;
      int sum8 = 0;
      pB2 = pB + heads * projections;
      pB3 = pB2 + heads * projections;
      pB4 = pB3 + heads * projections;
      pA = pInBuffer + (seq_out* projections * heads);
      pA2 = pA + projections * heads;
      for (proj_head_in = 0; proj_head_in < ((projections*heads)>>2); proj_head_in++)
      { 
        vecA = *((v4s*)pA);
        vecA2 = *((v4s*)pA2);
        vecB = *((v4s*)pB);
        vecB2 = *((v4s*)pB2);
        vecB3 = *((v4s*)pB3);
        vecB4 = *((v4s*)pB4);
        sum = SumDotp(vecA, vecB, sum);
        sum2 = SumDotp(vecA, vecB2, sum2);
        sum3 = SumDotp(vecA, vecB3, sum3);
        sum4 = SumDotp(vecA, vecB4, sum4);
        sum5 = SumDotp(vecA2, vecB, sum5);
        sum6 = SumDotp(vecA2, vecB2, sum6);
        sum7 = SumDotp(vecA2, vecB3, sum7);
        sum8 = SumDotp(vecA2, vecB4, sum8);
        pA+=4;
        pA2+=4;
        pB+=4;
        pB2+=4;
        pB3+=4;
        pB4+=4;
      }
      *pOut = clip8(sum);
      pOut++;
      *pOut = clip8(sum2);
      pOut++;
      *pOut = clip8(sum3);
      pOut++;
      *pOut = clip8(sum4);
      pOut++;
      *pOut2 = clip8(sum5);
      pOut2++;
      *pOut2 = clip8(sum6);
      pOut2++;
      *pOut2 = clip8(sum7);
      pOut2++;
      *pOut2 = clip8(sum8);
      pOut2++;
      pB = pB + (3 * dim_embedding);
    }
  }
  pi_cl_team_barrier(0);
}
