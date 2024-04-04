/* ----------------------------------------------------------------------
#
# File: pulp_nn_kernels.h
#
# Last edited: 29.08.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*/


void __attribute__ ((noinline)) linearQK_4x2_H(
  const int8_t *  pInBuffer,
  const int8_t *  pWeight,
  const int16_t * pBiasBuffer,
  int8_t *        pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  projections,
  const uint16_t  heads,
  const int16_t   requant_div,
  const int16_t   requant_mul
);

void __attribute__ ((noinline)) linearO_4x2_H(
  const int8_t *  pInBuffer,
  const int8_t *  pWeight,
  const int16_t * pBiasBuffer,
  int8_t *        pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  projections,
  const uint16_t  heads,
  const int16_t   requant_div,
  const int16_t   requant_mul
);

void __attribute__ ((noinline)) linearV_4x2_H(
  const int8_t *  pInBuffer,
  const int8_t *  pWeight,
  const int16_t * pBiasBuffer,
  int8_t *        pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  projections,
  const uint16_t  heads,
  const int16_t   requant_div,
  const int16_t   requant_mul
);

void __attribute__ ((noinline)) matmul_4x2_S(
  const int8_t *  pInBuffer,
  const int8_t *  pWeight,
  int8_t *        pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  projections,
  const uint16_t  heads,
  const int16_t   requant_div,
  const int16_t   requant_mul
);

void __attribute__ ((noinline)) matmulSoftmax_4x2_S(
  const int8_t *  pInBuffer,
  const int8_t *  pWeight,
  int8_t *        pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  projections,
  const uint16_t  heads,
  const int16_t   requant_div,
  const int16_t   requant_mul,  
  const int32_t   coeffA,
  const int32_t   coeffB,
  const int32_t   coeffC,
  const int32_t   log2,
  const uint32_t  n_levels
);

void __attribute__ ((noinline)) matmulSoftmax_4x2_H(
  const int8_t *  pInBuffer,
  const int8_t *  pWeight,
  int8_t *        pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  projections,
  const uint16_t  heads,
  const int16_t   requant_div,
  const int16_t   requant_mul,  
  const int32_t   coeffA,
  const int32_t   coeffB,
  const int32_t   coeffC,
  const int32_t   log2,
  const uint32_t  n_levels
);

void __attribute__ ((noinline)) matmulSoftmax_FWA_v1(
  const int8_t *  pInBuffer,
  const int8_t *  pWeight,
  const int16_t *  pBias,
  int8_t *        pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  heads,
  const int16_t   pre_proj_requant_div,
  const int16_t   pre_proj_requant_mul,
  const int16_t   post_proj_requant_div,
  const int16_t   post_proj_requant_mul,  
  const int32_t   coeffA,
  const int32_t   coeffB,
  const int32_t   coeffC,
  const int32_t   log2,
  const uint32_t  n_levels
);

void __attribute__ ((noinline)) matmulSoftmax_FWA_v2(
  const int8_t *  pInBuffer,
  const int8_t *  pWeight,
  const int16_t *  pBias,
  int8_t *        pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  heads,
  const int16_t   pre_proj_requant_div,
  const int16_t   pre_proj_requant_mul,
  const int16_t   post_proj_requant_div,
  const int16_t   post_proj_requant_mul,  
  const int32_t   coeffA,
  const int32_t   coeffB,
  const int32_t   coeffC,
  const int32_t   log2,
  const uint32_t  n_levels
);

void __attribute__ ((noinline)) matmulSoftmax_FWA_v3_S(
  const int8_t *  pInBuffer,
  const int8_t *  pWeight,
  const int16_t *  pBias,
  int8_t *        pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  heads,
  const int16_t   pre_proj_requant_div,
  const int16_t   pre_proj_requant_mul,
  const int16_t   post_proj_requant_div,
  const int16_t   post_proj_requant_mul,  
  const int32_t   coeffA,
  const int32_t   coeffB,
  const int32_t   coeffC,
  const int32_t   log2,
  const uint32_t  n_levels
);

void __attribute__ ((noinline)) matmulSoftmax_FWA_v3_H(
  const int8_t *  pInBuffer,
  const int8_t *  pWeight,
  const int16_t *  pBias,
  int8_t *        pOutBuffer,
  const uint16_t  dim_sequence,
  const uint16_t  dim_embedding,
  const uint16_t  heads,
  const int16_t   pre_proj_requant_div,
  const int16_t   pre_proj_requant_mul,
  const int16_t   post_proj_requant_div,
  const int16_t   post_proj_requant_mul,  
  const int32_t   coeffA,
  const int32_t   coeffB,
  const int32_t   coeffC,
  const int32_t   log2,
  const uint32_t  n_levels
);

void iSoftmax(
  int8_t *        pInBuffer,
  uint8_t *       pOutBuffer,
  const int32_t   rowDimension,
  const int32_t   coeffA,
  const int32_t   coeffB,
  const int32_t   coeffC,
  const int32_t   log2,
  const uint32_t  n_levels
);

void bnEmbedding (
    int8_t * Im_in, 
    int8_t * Im_out,            
    uint32_t div1,
    uint32_t mul1,     
    uint32_t add1,          
    int16_t * emb,      
    uint32_t mul2,             
    uint32_t add2,  
    uint32_t div2,           
    uint16_t  dim_im_in_h,
    uint16_t  dim_im_in_w,
    uint16_t  ch_im_in
);

void pulp_nn_linear_i8_i8_i8(
                        int8_t *pIn,
                        int16_t *pBias,
                        int8_t *pOut,
                        int8_t *pWeight,
                        int32_t *pKappa,
                        int32_t *pLambda,
                        uint16_t out_mult,
                        uint16_t out_shift,
                        uint16_t dim_vec,
                        uint16_t num_o_neurons,
                        uint8_t flag_relu,
                        uint8_t flag_batch_norm);

void pulp_nn_linear_u8_i8_i8(
                        uint8_t *pIn,
                        int8_t *pBias,
                        int8_t *pOut,
                        int8_t *pWeight,
                        int32_t *pKappa,
                        int32_t *pLambda,
                        uint16_t out_mult,
                        uint16_t out_shift,
                        uint16_t dim_vec,
                        uint16_t num_o_neurons,
                        uint8_t flag_relu,
                        uint8_t flag_batch_norm);