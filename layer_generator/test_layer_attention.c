/*
 * test_templateL2.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
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

#define PROFILING
#include "pulp_atten_kernels.h"
#include "stats.h"
#include "pulp.h"
#include "dory.h"
#include "layer_attention_q_input_reuse.h"
#include "layer_attention_k_input_reuse.h"
#include "layer_attention_v_weight_reuse.h"
#include "layer_matmul_softmax.h"
#include "layer_matmul.h"
PI_L2 int8_t input_L2[${dim_sequence * dim_embedding}] = {
  ${input_content}
};
PI_L2 int8_t W_q[${dim_embedding * projections * heads}] = {
  ${W_q_content}
};
PI_L2 int8_t W_k[${dim_embedding * projections * heads}] = {
  ${W_k_content}
};
PI_L2 int8_t W_v[${dim_embedding * projections * heads}] = {
  ${W_v_content}
};
PI_L2 int8_t W_out[${dim_embedding * projections * heads}] = {
  ${W_out_content}
};
% if check == True:
PI_L2 int8_t qq[${dim_sequence * projections * heads}] = {
  ${qq_content}
};
PI_L2 int8_t kk[${dim_sequence * projections * heads}] = {
  ${kk_content}
};
PI_L2 int8_t vv[${dim_sequence * projections * heads}] = {
  ${vv_content}
};
PI_L2 int8_t matmul1[${dim_sequence * dim_sequence * heads}] = {
  ${matmul1_content}
};
PI_L2 int8_t matmul2[${dim_sequence * projections * heads}] = {
  ${matmul2_content}
};
PI_L2 int8_t out_f[${dim_sequence * dim_embedding}] = {
  ${out_f_content}
};
% endif

PI_L2 int8_t q_L2[${dim_sequence * projections * heads}];
PI_L2 int8_t k_L2[${dim_sequence * projections * heads}];
PI_L2 int8_t v_L2[${dim_sequence * projections * heads}];
PI_L2 int8_t mat1_L2[${dim_sequence * dim_sequence * heads}];
PI_L2 int8_t mat2_L2[${dim_sequence * projections * heads}];
PI_L2 int8_t out_final[${dim_sequence * dim_embedding}];

uint16_t dim_sequence = ${dim_sequence};
uint16_t dim_embedding = ${dim_embedding};
uint16_t projections = ${projections};
uint16_t heads = ${heads};
int32_t L3_input = 0;
int32_t L3_output = 1;
int32_t L3_weights_internal = 2;
int32_t bypass_activations = 4;
int32_t ram = 5;
int32_t out_mult=1;
int32_t inmul1=1;
int32_t inmul2=1;
int32_t out_shift=1;
char *l1_buffer;

// on cluster
void cluster_main(void *arg) {
  int *real_arg = (int *) arg;
  l1_buffer = pmsis_l1_malloc((uint32_t) 44000);
  unsigned int args[13] = {L3_input,
    L3_output,
    L3_weights_internal,
    input_L2,
    bypass_activations,
    q_L2,
    W_q,
    l1_buffer,
    ram,
    out_mult,
    inmul1,
    inmul2, 
    out_shift};
  int cycles_total=0;
  int MACs_total=0;
  START_PROFILING()
  pi_cl_team_barrier(0);
  layer_attention_q_input_reuse(args);
  args[5] = k_L2;
  args[6] = W_k;
  layer_attention_k_input_reuse(args);
  args[5] = v_L2;
  args[6] = W_v;
  layer_attention_v_weight_reuse(args);
  pi_cl_team_barrier(0);
  STOP_PROFILING_NOPRINT()
  int perf_cyc =  pi_perf_read(PI_PERF_CYCLES); 
  int MACs = ${dim_sequence*dim_embedding*projections*heads*3};
  cycles_total+=perf_cyc;
  MACs_total+= MACs;
  float perf_MAC =  (float)MACs/perf_cyc;
  if (pi_core_id() == 0)
  {
    printf(" Linear q-k-v\n"); 
    printf(" MACs: %-11d,",MACs ); 
    printf(" cycles: %-11d,",perf_cyc );
    printf(" MAC/cycle: %-8f,",perf_MAC ); 
    printf(" n. of Cores: %d\n",NUM_CORES); 
  }          
  args[3] = q_L2;
  args[4] = k_L2;
  args[5] = mat1_L2;
  START_PROFILING()
  pi_cl_team_barrier(0);
  layer_matmul_softmax(args);
  pi_cl_team_barrier(0);
  STOP_PROFILING_NOPRINT()
  perf_cyc =  pi_perf_read(PI_PERF_CYCLES); 
  MACs = ${dim_sequence*dim_sequence*projections*heads};
  perf_MAC =  (float)MACs/perf_cyc;
  cycles_total+=perf_cyc;
  MACs_total+= MACs;
  if (pi_core_id() == 0)
  {
    printf(" Matmul1\n"); 
    printf(" MACs: %-11d,",MACs );  
    printf(" cycles: %-11d,",perf_cyc );
    printf(" MAC/cycle: %-8f,",perf_MAC ); 
    printf(" n. of Cores: %d\n",NUM_CORES); 
  }
  args[3] = mat1_L2;
  args[4] = v_L2;
  args[5] = mat2_L2;
  START_PROFILING()
  pi_cl_team_barrier(0);
  layer_matmul(args);
  pi_cl_team_barrier(0);
  STOP_PROFILING_NOPRINT()
  perf_cyc =  pi_perf_read(PI_PERF_CYCLES); 
  MACs = ${dim_sequence*dim_sequence*projections*heads};
  cycles_total+=perf_cyc;
  MACs_total+= MACs;
  perf_MAC =  (float)MACs/perf_cyc;
  if (pi_core_id() == 0)
  {
    printf(" Matmul2\n"); 
    printf(" MACs: %-11d,",MACs ); 
    printf(" cycles: %-11d,",perf_cyc );
    printf(" MAC/cycle: %-8f,",perf_MAC ); 
    printf(" n. of Cores: %d\n",NUM_CORES); 
  }
  args[3] = mat2_L2;
  args[5] = out_final;
  args[6] = W_out;
  START_PROFILING()
  pi_cl_team_barrier(0);
  layer_attention_out(args);
  pi_cl_team_barrier(0);
  STOP_PROFILING_NOPRINT()
  perf_cyc =  pi_perf_read(PI_PERF_CYCLES); 
  MACs = ${dim_sequence*dim_embedding*projections*heads};
  cycles_total+=perf_cyc;
  MACs_total+= MACs;
  perf_MAC =  (float)MACs/perf_cyc;
  if (pi_core_id() == 0)
  {
    printf(" Linear out\n"); 
    printf(" MACs: %-11d,",MACs ); 
    printf(" cycles: %-11d,",perf_cyc );
    printf(" MAC/cycle: %-8f,",perf_MAC ); 
    printf(" n. of Cores: %d\n",NUM_CORES); 
  }
  if (pi_core_id() == 0)
  {
    printf(" Total\n");  
    printf(" cycles: %-11d,",cycles_total );
    printf(" MAC/cycle: %-8f,",(float)MACs_total/cycles_total ); 
    printf(" n. of Cores: %d\n",NUM_CORES); 
  }
}

void pulp_parallel(void *arg)
{
  printf("Prova fork\n");
  rt_team_fork(NUM_CORES, (void *)cluster_main, arg);
}

// on fabric controller
int main () {

  int arg[1];
  arg[0] = (unsigned int) input_L2;
  int error_presence;
  error_presence = 0;
  rt_cluster_mount(1, 0, 0, NULL);
  rt_cluster_call(NULL, 0, pulp_parallel, arg, NULL,2048, 0, rt_nb_pe(), NULL);
  rt_cluster_mount(0, 0, 0, NULL);
  int checksum, checksum_true;
% if check == True:
  printf("Matrix Q: \n");
  checksum_true = 0; checksum = 0;
  for (int i=0; i<${dim_sequence * projections * heads}; i++) 
  {
    checksum+=*(q_L2+i);
    checksum_true+=*(qq+i);
  }
  printf("Checksum Computed %4d True %4d\n", checksum, checksum_true);
  printf("Matrix K: \n");
  checksum_true = 0; checksum = 0;
  for (int i=0; i<${dim_sequence * projections * heads}; i++) 
  {
    checksum+=*(k_L2+i);
    checksum_true+=*(kk+i);
  }
  printf("Checksum Computed %4d True %4d\n", checksum, checksum_true);
  printf("Matrix V: \n");
  checksum_true = 0; checksum = 0;
  for (int i=0; i<${dim_sequence * projections * heads}; i++) 
  {
    checksum+=*(v_L2+i);
    checksum_true+=*(vv+i);
  }
  printf("Checksum Computed %4d True %4d\n", checksum, checksum_true);
  printf("Matrix attention: \n");
  checksum_true = 0; checksum = 0;
  for (int i=0; i<${dim_sequence * dim_sequence * heads}; i++) 
  {
    checksum+=*(mat1_L2+i);
    checksum_true+=*(matmul1+i);
  }
  printf("Checksum Computed %4d True %4d\n", checksum, checksum_true);
  printf("Matrix final: \n");
  checksum_true = 0; checksum = 0;
  for (int i=0; i<${dim_sequence * projections * heads}; i++) 
  {
    checksum+=*(mat2_L2+i);
    checksum_true+=*(matmul2+i);
  }
  printf("Checksum Computed %4d True %4d\n", checksum, checksum_true);
  printf("Linear final: \n");
  checksum_true = 0; checksum = 0;
  for (int i=0; i<${dim_sequence * dim_sequence}; i++) 
  {
    checksum+=*(out_final+i);
    checksum_true+=*(out_f+i);
  }
  printf("Checksum Computed %4d True %4d\n", checksum, checksum_true);
  printf("End Layer :\tOk\n");
% endif

% if check_byte == True:
  printf("Matrix Q: \n");
  for (int i=0; i<${dim_sequence * projections * heads}; i++) 
  {
    if (*(q_L2+i) != *(qq+i))
      printf("Value %4d Correct %4d\n", *(q_L2+i),  *(qq+i));
  }
  printf("Matrix K: \n");
  for (int i=0; i<${dim_sequence * projections * heads}; i++) 
  {
    if (*(k_L2+i) != *(kk+i))
      printf("Value %4d Correct %4d\n", *(k_L2+i),  *(kk+i));
  }
  printf("Matrix V: \n");
  for (int i=0; i<${dim_sequence * projections * heads}; i++) 
  {
    if (*(v_L2+i) != *(vv+i))
      printf("Value %d Predicted %4d Correct %4d\n", i, *(v_L2+i),  *(vv+i));
  }
  printf("Matrix attention: \n");
  for (int i=0; i<${dim_sequence * dim_sequence * heads}; i++) 
  {
    if (*(mat1_L2+i) != *(matmul1+i))
      printf("Value %d Predicted %4d Correct %4d\n", i, *(mat1_L2+i),  *(matmul1+i));
  }
  printf("Matrix final: \n");
  for (int i=0; i<${dim_sequence * dim_sequence * heads}; i++) 
  {
    if (*(mat2_L2+i) != *(matmul2+i))
      printf("Value %d Predicted %4d Correct %4d\n", i, *(mat2_L2+i),  *(matmul2+i));
  }
% endif


}
