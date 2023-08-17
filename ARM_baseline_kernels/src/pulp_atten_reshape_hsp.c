/*
 * pulp_atten_reshape_hsp.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 *
 * Copyright (C) 2018-2020 University of Bologna
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

pulp_atten_reshape_hsp(
      q7_t *pInBuffer,
      q7_t *pOutBuffer,
      const uint16_t  dim_sequence,
      const uint16_t  projections,
      const uint16_t  heads
)
{
  // local vars
  int head_out, proj_out, seq_out;
  int8_t *pOut = pOutBuffer;
  int8_t *pIn = pInBuffer;
  for (head_out = 0; head_out < heads; head_out++)
  {
    for (proj_out = 0; proj_out < projections; proj_out++)
    {  
      for (seq_out = 0; seq_out < dim_sequence; seq_out++)
      {
        *pOut = *(pIn + proj_out + head_out*projections + seq_out*projections*heads);
        pOut+=1;
      }
    }
  }
}
