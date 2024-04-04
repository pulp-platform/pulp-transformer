#ifndef __PROF_STATS__
#define __PROF_STATS__
#endif

#ifdef PROFILING

#define START_PROFILING() \
	if (pi_core_id()==0) { \
    pi_perf_conf((1<<PI_PERF_CYCLES));			\
    pi_perf_reset();						\
    pi_perf_stop();						\
    pi_perf_start(); \
	}

#define STOP_PROFILING() \
	if (pi_core_id()==0) { \
	pi_perf_stop();			\
	int cid = pi_core_id();					\
	printf("[%d] : num_cycles: %d\n",cid,pi_perf_read(PI_PERF_CYCLES) ); \
	}

#define STOP_PROFILING_NOPRINT() \
	if (pi_core_id()==0) { \
    pi_perf_stop();           \
	}
#else
#define START_PROFILING()
#define STOP_PROFILING()
#define STOP_PROFILING_NOPRINT()
#endif

