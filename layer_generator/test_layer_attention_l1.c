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

PI_L1 int8_t input[${dim_sequence * dim_embedding}] = {
  ${input_content}
};
PI_L1 int8_t W_q[${dim_embedding * projections * heads}] = {
  ${W_q_content}
};
PI_L1 int8_t W_k[${dim_embedding * projections * heads}] = {
  ${W_k_content}
};
PI_L1 int8_t W_v[${dim_embedding * projections * heads}] = {
  ${W_v_content}
};
PI_L1 int8_t W_out[${dim_embedding * projections * heads}] = {
  ${W_out_content}
};
% if check == True:
PI_L1 int8_t qq[${dim_sequence * projections * heads}] = {
  ${qq_content}
};
PI_L1 int8_t kk[${dim_sequence * projections * heads}] = {
  ${kk_content}
};
PI_L1 int8_t vv[${dim_sequence * projections * heads}] = {
  ${vv_content}
};
PI_L1 int8_t matmul1[${dim_sequence * dim_sequence * heads}] = {
  ${matmul1_content}
};
PI_L1 int8_t matmul2[${dim_sequence * projections * heads}] = {
  ${matmul2_content}
PI_L2 int8_t out_f[${dim_sequence * dim_embedding}] = {
  ${out_f_content}
};
};
% endif
PI_L1 int8_t q[${dim_sequence * projections * heads}];
PI_L1 int8_t support_q[${dim_sequence * projections * heads}];
PI_L1 int8_t k[${dim_sequence * projections * heads}];
PI_L1 int8_t support_k[${dim_sequence * projections * heads}];
PI_L1 int8_t v[${dim_sequence * projections * heads}];
PI_L1 int8_t support_v[${dim_sequence * projections * heads}];
PI_L1 int8_t mat1[${dim_sequence * dim_sequence * heads}];
PI_L1 int8_t mat2[${dim_sequence * projections * heads}];
PI_L1 int8_t support_mat2[${dim_sequence * projections * heads}];
PI_L1 int8_t out_final[${dim_sequence * dim_embedding}];

uint16_t dim_sequence = ${dim_sequence};
uint16_t dim_embedding = ${dim_embedding};
uint16_t projections = ${projections};
uint16_t heads = ${heads};

// on cluster
void cluster_main(void *arg) {
  int *real_arg = (int *) arg;
  int cycles_total=0;
  int MACs_total=0;
  START_PROFILING()
  pi_cl_team_barrier(0);
  pulp_atten_linear_4x2_input_reuse((unsigned int) real_arg[0],
                    (unsigned int) real_arg[1],
                    (unsigned int) real_arg[4],
                    dim_sequence,
                    dim_embedding,
                    projections,
                    heads);
  pi_cl_team_barrier(0);
  STOP_PROFILING_NOPRINT()
  int perf_cyc =  pi_perf_read(PI_PERF_CYCLES); 
  int MACs = ${dim_sequence*dim_embedding*projections*heads};
  cycles_total+=perf_cyc;
  MACs_total+= MACs;
  float perf_MAC =  (float)MACs/perf_cyc;
  if (pi_core_id() == 0)
  {
    printf("Linear q\n"); 
    printf(" MACs: %-11d,",MACs ); 
    printf(" cycles: %-11d,",cycles_total );
    printf(" MAC/cycle: %-8f,",perf_MAC ); 
    printf(" n. of Cores: %d\n",NUM_CORES); 
  }          
  START_PROFILING()
  pi_cl_team_barrier(0);
  pulp_atten_linear_4x2_input_reuse((unsigned int) real_arg[0],
                                (unsigned int) real_arg[2],
                                (unsigned int) real_arg[5],
                                dim_sequence,
                                dim_embedding,
                                projections,
                                heads);

  pi_cl_team_barrier(0);
  STOP_PROFILING_NOPRINT()
  perf_cyc =  pi_perf_read(PI_PERF_CYCLES); 
  MACs = ${dim_sequence*dim_embedding*projections*heads};
  cycles_total+=perf_cyc;
  MACs_total+= MACs;
  perf_MAC =  (float)MACs/perf_cyc;
  if (pi_core_id() == 0)
  {
    printf("Linear k\n"); 
    printf(" MACs: %-11d,",MACs ); 
    printf(" cycles: %-11d,",cycles_total );
    printf(" MAC/cycle: %-8f,",perf_MAC ); 
    printf(" n. of Cores: %d\n",NUM_CORES); 
  }
  START_PROFILING()
  pi_cl_team_barrier(0);
  pulp_atten_linear_4x2_weights_reuse((unsigned int) real_arg[0],
                                (unsigned int) real_arg[3],
                                (unsigned int) real_arg[6],
                                dim_sequence,
                                dim_embedding,
                                projections,
                                heads);
  pi_cl_team_barrier(0);
  STOP_PROFILING_NOPRINT()
  perf_cyc =  pi_perf_read(PI_PERF_CYCLES); 
  MACs = ${dim_sequence*dim_embedding*projections*heads};
  cycles_total+=perf_cyc;
  MACs_total+= MACs;
  perf_MAC =  (float)MACs/perf_cyc;
  if (pi_core_id() == 0)
  {
    printf("Linear v\n"); 
    printf(" MACs: %-11d,",MACs ); 
    printf(" cycles: %-11d,",cycles_total );
    printf(" MAC/cycle: %-8f,",perf_MAC ); 
    printf(" n. of Cores: %d\n",NUM_CORES); 
  }
  START_PROFILING()
  pi_cl_team_barrier(0);
  pulp_atten_matmul_4x2_strided((unsigned int) real_arg[4],
                                (unsigned int) real_arg[5],
                                (unsigned int) real_arg[7],
                                dim_sequence,
                                projections,
                                heads);
  pi_cl_team_barrier(0);
  STOP_PROFILING_NOPRINT()
  perf_cyc =  pi_perf_read(PI_PERF_CYCLES); 
  MACs = ${dim_sequence*dim_sequence*projections*heads};
  cycles_total+=perf_cyc;
  MACs_total+= MACs;
  perf_MAC =  (float)MACs/perf_cyc;
  if (pi_core_id() == 0)
  {
    printf("Matmul1\n"); 
    printf(" MACs: %-11d,",MACs );
    printf(" cycles: %-11d,",cycles_total ); 
    printf(" MAC/cycle: %-8f,",perf_MAC ); 
    printf(" n. of Cores: %d\n",NUM_CORES); 
  }
  START_PROFILING()
  pi_cl_team_barrier(0);
  pulp_atten_matmul_4x2_sequential((unsigned int) real_arg[7],
                                (unsigned int) real_arg[6],
                                (unsigned int) real_arg[12],
                                dim_sequence,
                                projections,
                                heads);
  pi_cl_team_barrier(0);
  STOP_PROFILING_NOPRINT()
  perf_cyc =  pi_perf_read(PI_PERF_CYCLES); 
  MACs = ${dim_sequence*dim_sequence*projections*heads};
  cycles_total+=perf_cyc;
  MACs_total+= MACs;
  perf_MAC =  (float)MACs/perf_cyc;
  if (pi_core_id() == 0)
  {
    printf("Matmul2\n"); 
    printf(" MACs: %-11d,",MACs ); 
    printf(" cycles: %-11d,",cycles_total );
    printf(" MAC/cycle: %-8f,",perf_MAC ); 
    printf(" n. of Cores: %d\n",NUM_CORES); 
  }
  START_PROFILING()
  pi_cl_team_barrier(0);
  pulp_atten_linear_4x2_out((unsigned int) real_arg[12],
                                (unsigned int) real_arg[3],
                                (unsigned int) real_arg[0],
                                dim_sequence,
                                dim_embedding,
                                projections,
                                heads);
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

  int arg[13];
  arg[0] = (unsigned int) input;
  arg[1] = (unsigned int) W_q;
  arg[2] = (unsigned int) W_k;
  arg[3] = (unsigned int) W_v;
  arg[4] = (unsigned int) q;
  arg[5] = (unsigned int) k;
  arg[6] = (unsigned int) v;
  arg[7] = (unsigned int) mat1;
  arg[8] = (unsigned int) mat2;
  arg[9] = (unsigned int) support_q;
  arg[10] = (unsigned int) support_k;
  arg[11] = (unsigned int) support_v;
  arg[12] = (unsigned int) support_mat2;
  int error_presence;
  error_presence = 0;
  rt_cluster_mount(1, 0, 0, NULL);
  rt_cluster_call(NULL, 0, pulp_parallel, arg, NULL,1024, 0, rt_nb_pe(), NULL);
  rt_cluster_mount(0, 0, 0, NULL);
  int checksum, checksum_true;
% if check == True:
  printf("Matrix Q: \n");
  checksum_true = 0; checksum = 0;
  for (int i=0; i<${dim_sequence * projections * heads}; i++) 
  {
    checksum+=*(q+i);
    checksum_true+=*(qq+i);
  }
  printf("Checksum Computed %4d True %4d\n", checksum, checksum_true);
  printf("Matrix K: \n");
  checksum_true = 0; checksum = 0;
  for (int i=0; i<${dim_sequence * projections * heads}; i++) 
  {
    checksum+=*(k+i);
    checksum_true+=*(kk+i);
  }
  printf("Checksum Computed %4d True %4d\n", checksum, checksum_true);
  printf("Matrix V: \n");
  checksum_true = 0; checksum = 0;
  for (int i=0; i<${dim_sequence * projections * heads}; i++) 
  {
    checksum+=*(v+i);
    checksum_true+=*(vv+i);
  }
  printf("Checksum Computed %4d True %4d\n", checksum, checksum_true);
  printf("Matrix attention: \n");
  checksum_true = 0; checksum = 0;
  for (int i=0; i<${dim_sequence * dim_sequence * heads}; i++) 
  {
    checksum+=*(mat1+i);
    checksum_true+=*(matmul1+i);
  }
  printf("Checksum Computed %4d True %4d\n", checksum, checksum_true);
  printf("Matrix final: \n");
  checksum_true = 0; checksum = 0;
  for (int i=0; i<${dim_sequence * projections * heads}; i++) 
  {
    checksum+=*(mat2+i);
    checksum_true+=*(matmul2+i);
  }
  printf("Checksum Computed %4d True %4d\n", checksum, checksum_true);
  printf("End Layer :\tOk\n");
% endif

% if check_byte == True:
  printf("Matrix Q: \n");
  for (int i=0; i<${dim_sequence * projections * heads}; i++) 
  {
    if (*(q+i) != *(qq+i))
      printf("Value %4d Correct %4d\n", *(q+i),  *(qq+i));
  }
  printf("Matrix K: \n");
  for (int i=0; i<${dim_sequence * projections * heads}; i++) 
  {
    if (*(k+i) != *(kk+i))
      printf("Value %4d Correct %4d\n", *(k+i),  *(kk+i));
  }
  printf("Matrix V: \n");
  for (int i=0; i<${dim_sequence * projections * heads}; i++) 
  {
    if (*(v+i) != *(vv+i))
      printf("Value %d Predicted %4d Correct %4d\n", i, *(v+i),  *(vv+i));
  }
  printf("Matrix attention: \n");
  for (int i=0; i<${dim_sequence * dim_sequence * heads}; i++) 
  {
    if (*(mat1+i) != *(matmul1+i))
      printf("Value %d Predicted %4d Correct %4d\n", i, *(mat1+i),  *(matmul1+i));
  }
  printf("Matrix final: \n");
  for (int i=0; i<${dim_sequence * dim_sequence * heads}; i++) 
  {
    if (*(mat2+i) != *(matmul2+i))
      printf("Value %d Predicted %4d Correct %4d\n", i, *(mat2+i),  *(matmul2+i));
  }
% endif


}
