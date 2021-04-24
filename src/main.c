/**
 *******************************************************************************
 * @file main.c
 * @version 1.0.1
 * @date 2020-09-01
 * @copyright Copyright © 2020 by University of Luxembourg.
 * @author Developed at SnT APSIA by: Hao Cheng, Johann Groszschaedl and Jiaqi Tian.
 *
 * @brief C source file of tests and benchmarks. 
 *
 * @details 
 * This file contains test and benchmark functions.
 *******************************************************************************
 */

#include "gfparith.h"
#include "moncurve.h"
#include "tedcurve.h"
#include "ecdh.h"
#include "utils.h"
#include <time.h>
#include <string.h>

/**
 * @brief Test the correctness of our software.
 *
 * @details
 * Test the correctness of key generation and shared secret according to the 
 * test vectors of RFC 7748 and random private keys.
 */
void test_ecdh()
{
  uint32_t pk_a[NWORDS], pk_b[NWORDS], pk_c[NWORDS], pk_d[NWORDS];
  uint32_t ss_a[NWORDS], ss_b[NWORDS], ss_c[NWORDS], ss_d[NWORDS], r[NWORDS];
  __m256i sk[8], pk[NWORDS], ss[NWORDS];
  int i, j, seed, wrong = 0;

  // initialize random generator
  seed = (int)time(NULL);
	srandom(seed);

  // Alice's private key (from RFC7748)
  uint32_t sk_a[8] = { 0x0a6d0777, 0x7da51873, 0x72c1163c, 0x4566b251, \
    0x872f4cdf, 0x2a99c0eb, 0xa5fb77b1, 0x2a2cb91d};
  // Bob's private key (from RFC7748)
  uint32_t sk_b[8] = { 0x7e08ab5d, 0x4b8a4a62, 0x8b7fe179, 0xe60e8083, \
    0x29b13b6f, 0xfdb61826, 0x278b2f1c, 0xebe088ff};
  // Carol's private key (random)
  uint32_t sk_c[8] = { 
    (uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random(),
    (uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random()};
  // Dave's private key (random)
  uint32_t sk_d[8] = { 
    (uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random(),
    (uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random()};

  puts("\n*******************************************************************");
  puts("CORRECTNESS TEST:");
  puts("-------------------------------------------------------------------");
  puts("FOUR instances in each run of the program.");
  puts("1st: Alice;    2nd: Bob;  (Alice & Bob are test vectors of RFC7748)");
  puts("3rd: Carol;    4th: Dave. (Carol & Dave randomly each time)\n");
  puts("We can test our software by assuming two pairs of them are sharing the secret:");
  puts("Alice  <---------------------------->  Bob ");
  puts("Carol  <---------------------------->  Dave");

  puts("\n* Private key:");
  mpi29_print("  - Alice : ", sk_a, 8);
  mpi29_print("  - Bob   : ", sk_b, 8);
  mpi29_print("  - Carol : ", sk_c, 8);
  mpi29_print("  - Dave  : ", sk_d, 8);

  // intialize AVX2 vector of private key
  for (i = 0; i < 8; i++) sk[i] = VSET64(sk_d[i], sk_c[i], sk_b[i], sk_a[i]);
  
  keygen(pk, sk);

  for (i = 0; i < NWORDS; i++) {
    pk_a[i] = VEXTR32(pk[i], 0);
    pk_b[i] = VEXTR32(pk[i], 2);
    pk_c[i] = VEXTR32(pk[i], 4);
    pk_d[i] = VEXTR32(pk[i], 6);
  }

  puts("\n* Public key:");
  mpi29_conv_29to32(r, pk_a, NWORDS, NWORDS);
  mpi29_print("  - Alice : ", r, 8);
  mpi29_conv_29to32(r, pk_b, NWORDS, NWORDS);
  mpi29_print("  - Bob   : ", r, 8);
  mpi29_conv_29to32(r,  pk_c, NWORDS, NWORDS);
  mpi29_print("  - Carol : ", r, 8);
  mpi29_conv_29to32(r, pk_d, NWORDS, NWORDS);
  mpi29_print("  - Dave  : ", r, 8);

  // We simply swap the position of (pk_a and pk_b) and (pk_c and pk_d) 
  // in the AVX2 vectors. So now it is (pk_c, pk_d, pk_a, pk_b)
  for (i = 0; i < NWORDS; i++) pk[i] = VPERM64(pk[i], 0xB1);

  // * Shared Secret
  // sk_m (sk_d, sk_c, sk_b, sk_a)
  // pk_m (pk_c, pk_d, pk_a, pk_b)
  // each pair should obtain the same shared secret
  sharedsecret(ss, sk, pk);

  puts("\n* Shared secret:");
  // 1st instance 
  for (i = 0; i < NWORDS; i++) ss_a[i] = VEXTR32(ss[i], 0);
  mpi29_conv_29to32(r, ss_a, NWORDS, NWORDS);
  mpi29_print("  - Alice : ", r, 8);

  // 2nd instance 
  for (i = 0; i < NWORDS; i++) ss_b[i] = VEXTR32(ss[i], 2);
  mpi29_conv_29to32(r, ss_b, NWORDS, NWORDS);
  mpi29_print("  - Bob   : ", r, 8);

  // 3rd instance 
  for (i = 0; i < NWORDS; i++) ss_c[i] = VEXTR32(ss[i], 4);
  mpi29_conv_29to32(r, ss_c, NWORDS, NWORDS);
  mpi29_print("  - Carol : ", r, 8);

  // 4th instance 
  for (i = 0; i < NWORDS; i++) ss_d[i] = VEXTR32(ss[i], 6);
  mpi29_conv_29to32(r, ss_d, NWORDS, NWORDS);
  mpi29_print("  - Dave  : ", r, 8);

  puts("\n* Correctness:");
  if (memcmp(ss_a, ss_b, NWORDS*sizeof(uint32_t)))
        printf("Shared secret between Alice and Bob  : \x1b[31mNOT EQUAL!\x1b[0m\n");
  else
    printf("Shared secret between Alice and Bob  : \x1b[32mEQUAL!\x1b[0m\n");

  if (memcmp(ss_c, ss_d, NWORDS*sizeof(uint32_t)))
    printf("Shared secret between Carol and Dave : \x1b[31mNOT EQUAL!\x1b[0m\n");
  else
    printf("Shared secret between Carol and Dave : \x1b[32mEQUAL!\x1b[0m\n");

  puts("-------------------------------------------------------------------");

  puts("Test ECDH for 1000 times (randomly each time):");

  for (j = 0; j < 1000; j++) {
    for (i = 0; i < 8; i++) {
      sk_a[i] = (uint32_t)random();
      sk_b[i] = (uint32_t)random();
      sk_c[i] = (uint32_t)random();
      sk_d[i] = (uint32_t)random();
    }

    // intialize AVX2 vector of private key
    for (i = 0; i < 8; i++) sk[i] = VSET64(sk_d[i], sk_c[i], sk_b[i], sk_a[i]);
    
    keygen(pk, sk);

    // we simply swap the position of pk_a and pk_b in the AVX2 vectors
    // so now it is (pk_c, pk_d, pk_a, pk_b)
    for (i = 0; i < NWORDS; i++) pk[i] = VPERM64(pk[i], 0xB1);

    // * Shared Secret
    // sk_m (sk_d, sk_c, sk_b, sk_a)
    // pk_m (pk_c, pk_d, pk_a, pk_b)
    // each pair should obtain the same shared secret
    sharedsecret(ss, sk, pk);

    for (int i = 0; i < NWORDS; i++) {
      ss_a[i] = VEXTR32(ss[i], 0);
      ss_b[i] = VEXTR32(ss[i], 2);
      ss_c[i] = VEXTR32(ss[i], 4);
      ss_d[i] = VEXTR32(ss[i], 6);
    }
    wrong = wrong | (memcmp(ss_a, ss_b, NWORDS*sizeof(uint32_t)));
    wrong = wrong | (memcmp(ss_c, ss_d, NWORDS*sizeof(uint32_t)));
  }

  // if wrong != 0 means there is at least one test that is wrong!
  // if wrong == 0 means that all the shared secrets are equal!
  if (wrong) 
    printf("TEST: \x1b[31mNOT PASS!\x1b[0m\n");
  else 
    printf("TEST : \x1b[32mPASS!\x1b[0m\n");

  puts("*******************************************************************");
}

/**
 * @brief Measure latency of field operations.
 *
 * @details
 * Measure latency of field addition, subtraction, multiplication and squaring.
 */
void timing_fp_arith()
{
  __m256i a[NWORDS], b[NWORDS], r[NWORDS];
  int i, seed;

  // initialize random generator
  seed = (int)time(NULL);
	srandom(seed);

  // randomize the input
  for (i = 0; i < NWORDS; i++) {
    a[i] = VSET64((uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random());
    b[i] = VSET64((uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random());
    r[i] = VSET64((uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random());
  }

  // benchmark part
  uint64_t start_cycles, end_cycles, diff_cycles;
  int iterations = 1000000;

  puts("");

  // load cache
  for (i = 0; i < iterations; i++) mpi29_gfp_add_avx2(r, r, b);
  // measure timing
  start_cycles = read_tsc();
  for (i = 0; i < iterations; i++) {
    mpi29_gfp_add_avx2(r, r, b);
    mpi29_gfp_add_avx2(r, r, b);
    mpi29_gfp_add_avx2(r, r, b);
    mpi29_gfp_add_avx2(r, r, b);
    mpi29_gfp_add_avx2(r, r, b);
    mpi29_gfp_add_avx2(r, r, b);
    mpi29_gfp_add_avx2(r, r, b);
    mpi29_gfp_add_avx2(r, r, b);
    mpi29_gfp_add_avx2(r, r, b);
    mpi29_gfp_add_avx2(r, r, b);
  }
  end_cycles = read_tsc();
  diff_cycles = (end_cycles-start_cycles)/(10*iterations);
  printf("* 4-Way ADD: %lld\n", diff_cycles);

  // load cache
  for (i = 0; i < iterations; i++) mpi29_gfp_sub_avx2(r, r, a);
  // measure timing
  start_cycles = read_tsc();
  for (i = 0; i < iterations; i++) {
    mpi29_gfp_sub_avx2(r, r, a);
    mpi29_gfp_sub_avx2(r, r, a);
    mpi29_gfp_sub_avx2(r, r, a);
    mpi29_gfp_sub_avx2(r, r, a);
    mpi29_gfp_sub_avx2(r, r, a);
    mpi29_gfp_sub_avx2(r, r, a);
    mpi29_gfp_sub_avx2(r, r, a);
    mpi29_gfp_sub_avx2(r, r, a);
    mpi29_gfp_sub_avx2(r, r, a);
    mpi29_gfp_sub_avx2(r, r, a);
  }
  end_cycles = read_tsc();
  diff_cycles = (end_cycles-start_cycles)/(10*iterations);
  printf("* 4-Way SUB: %lld\n", diff_cycles);

  // load cache
  for (i = 0; i < iterations; i++) mpi29_gfp_sbc_avx2(r, r, b);
  // measure timing
  start_cycles = read_tsc();
  for (i = 0; i < iterations; i++) {
    mpi29_gfp_sbc_avx2(r, r, b);
    mpi29_gfp_sbc_avx2(r, r, b);
    mpi29_gfp_sbc_avx2(r, r, b);
    mpi29_gfp_sbc_avx2(r, r, b);
    mpi29_gfp_sbc_avx2(r, r, b);
    mpi29_gfp_sbc_avx2(r, r, b);
    mpi29_gfp_sbc_avx2(r, r, b);
    mpi29_gfp_sbc_avx2(r, r, b);
    mpi29_gfp_sbc_avx2(r, r, b);
    mpi29_gfp_sbc_avx2(r, r, b);
  }
  end_cycles = read_tsc();
  diff_cycles = (end_cycles-start_cycles)/(10*iterations);
  printf("* 4-Way SBC: %lld\n", diff_cycles);

  // load cache
  for (i = 0; i < iterations; i++) mpi29_gfp_mul_avx2(r, r, a);
  // measure timing
  start_cycles = read_tsc();
  for (i = 0; i < iterations; i++) {
    mpi29_gfp_mul_avx2(r, r, a);
    mpi29_gfp_mul_avx2(r, r, a);
    mpi29_gfp_mul_avx2(r, r, a);
    mpi29_gfp_mul_avx2(r, r, a);
    mpi29_gfp_mul_avx2(r, r, a);
    mpi29_gfp_mul_avx2(r, r, a);
    mpi29_gfp_mul_avx2(r, r, a);
    mpi29_gfp_mul_avx2(r, r, a);
    mpi29_gfp_mul_avx2(r, r, a);
    mpi29_gfp_mul_avx2(r, r, a);
  }
  end_cycles = read_tsc();
  diff_cycles = (end_cycles-start_cycles)/(10*iterations);
  printf("* 4-Way MUL: %lld\n", diff_cycles);

  // load cache
  for (i = 0; i < iterations; i++) mpi29_gfp_sqr_avx2(r, r);
  // measure timing 
  start_cycles = read_tsc();
  for (i = 0; i < iterations; i++) {
    mpi29_gfp_sqr_avx2(r, r);
    mpi29_gfp_sqr_avx2(r, r);
    mpi29_gfp_sqr_avx2(r, r);
    mpi29_gfp_sqr_avx2(r, r);
    mpi29_gfp_sqr_avx2(r, r);
    mpi29_gfp_sqr_avx2(r, r);
    mpi29_gfp_sqr_avx2(r, r);
    mpi29_gfp_sqr_avx2(r, r);
    mpi29_gfp_sqr_avx2(r, r);
    mpi29_gfp_sqr_avx2(r, r);
  }
  end_cycles = read_tsc();
  diff_cycles = (end_cycles-start_cycles)/(10*iterations);
  printf("* 4-Way SQR: %lld\n", diff_cycles);
}

/**
 * @brief Measure latency of point operations.
 *
 * @details
 * Measure latency of ladder-step on Montgomery curve and point addition, 
 * doubling, table-based multiplication on twisted Edward curve.
 */
void timing_point_arith()
{
  ProPoint p, q;
  ExtPoint r, a;
  __m256i t[NWORDS];
  int i, seed;

  // initialize random generator
  seed = (int)time(NULL);
	srandom(seed);

  // randomize the input
  for (i = 0; i < NWORDS; i++) {
    t[i]   = VSET64((uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random());
    p.x[i] = VSET64((uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random());
    p.y[i] = VSET64((uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random());
    p.z[i] = VSET64((uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random());
    q.x[i] = VSET64((uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random());
    q.y[i] = VSET64((uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random());
    q.z[i] = VSET64((uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random());
    r.x[i] = VSET64((uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random());
    r.y[i] = VSET64((uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random());
    r.z[i] = VSET64((uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random());
    r.e[i] = VSET64((uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random());
    r.h[i] = VSET64((uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random());
    a.x[i] = VSET64((uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random());
    a.y[i] = VSET64((uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random());
    a.z[i] = VSET64((uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random());
    a.e[i] = VSET64((uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random());
    a.h[i] = VSET64((uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random());
  }

  // benchmark part
  uint64_t start_cycles, end_cycles, diff_cycles;
  int iterations = 100000;

  puts("\nMontgomery curve:");

  // load cache
  for (i = 0; i < iterations; i++) mon_ladder_step_avx2(&p, &q, t);
  // measure timing
  start_cycles = read_tsc();
  for (i = 0; i < iterations; i++) {
    mon_ladder_step_avx2(&p, &q, t);
    mon_ladder_step_avx2(&p, &q, t);
    mon_ladder_step_avx2(&p, &q, t);
    mon_ladder_step_avx2(&p, &q, t);
    mon_ladder_step_avx2(&p, &q, t);
    mon_ladder_step_avx2(&p, &q, t);
    mon_ladder_step_avx2(&p, &q, t);
    mon_ladder_step_avx2(&p, &q, t);
    mon_ladder_step_avx2(&p, &q, t);
    mon_ladder_step_avx2(&p, &q, t);
  }
  end_cycles = read_tsc();
  diff_cycles = (end_cycles-start_cycles)/(10*iterations);
  printf("* 4-Way Ladder-Step: %lld\n", diff_cycles);

  puts("\ntwisted Edwards curve:");

  // load cache
  for (i = 0; i < iterations; i++) ted_point_add_avx2(&r, &a, &p);
  start_cycles = read_tsc();
  for (i = 0; i < iterations; i++) {
    ted_point_add_avx2(&r, &r, &p);
    ted_point_add_avx2(&r, &r, &p);
    ted_point_add_avx2(&r, &r, &p);
    ted_point_add_avx2(&r, &r, &p);
    ted_point_add_avx2(&r, &r, &p);
    ted_point_add_avx2(&r, &r, &p);
    ted_point_add_avx2(&r, &r, &p);
    ted_point_add_avx2(&r, &r, &p);
    ted_point_add_avx2(&r, &r, &p);
    ted_point_add_avx2(&r, &r, &p);
  }
  end_cycles = read_tsc();
  diff_cycles = (end_cycles-start_cycles)/(10*iterations);
  printf("* 4-Way Point Addition: %lld\n", diff_cycles);

  // load cache
  for (i = 0; i < iterations; i++) ted_point_dbl_avx2(&r, &a);
  start_cycles = read_tsc();
  for (i = 0; i < iterations; i++) {
    ted_point_dbl_avx2(&r, &r);
    ted_point_dbl_avx2(&r, &r);
    ted_point_dbl_avx2(&r, &r);
    ted_point_dbl_avx2(&r, &r);
    ted_point_dbl_avx2(&r, &r);
    ted_point_dbl_avx2(&r, &r);
    ted_point_dbl_avx2(&r, &r);
    ted_point_dbl_avx2(&r, &r);
    ted_point_dbl_avx2(&r, &r);
    ted_point_dbl_avx2(&r, &r);
  }
  end_cycles = read_tsc();
  diff_cycles = (end_cycles-start_cycles)/(10*iterations);
  printf("* 4-Way Point Doubling: %lld\n", diff_cycles);

  // load cache
  for (i = 0; i < iterations; i++) ted_point_query_table_avx2(&p, i&0x1F, VSET164(1));
  start_cycles = read_tsc();
  for (i = 0; i < iterations; i++)
  {
    ted_point_query_table_avx2(&p, i&0x1F, t[i%9]);
    ted_point_query_table_avx2(&p, i&0x1F, t[i%9+1]);
    ted_point_query_table_avx2(&p, i&0x1F, t[i%9+2]);
    ted_point_query_table_avx2(&p, i&0x1F, t[i%9+3]);
    ted_point_query_table_avx2(&p, i&0x1F, t[i%9+4]);
    ted_point_query_table_avx2(&p, i&0x1F, t[i%9+5]);
    ted_point_query_table_avx2(&p, i&0x1F, t[i%9+6]);
    ted_point_query_table_avx2(&p, i&0x1F, t[i%9+7]);
    ted_point_query_table_avx2(&p, i&0x1F, t[i%9+8]);
    ted_point_query_table_avx2(&p, i&0x1F, t[i%9]);
  } 
  end_cycles = read_tsc();
  diff_cycles = (end_cycles-start_cycles)/(10*iterations);
  printf("* 4-Way Table Query   : %lld\n", diff_cycles);

}

/**
 * @brief Measure latency of Diffie-Hellman functions.
 *
 * @details
 * Measure latency of key generation and shared secret.
 */
void timing_ecdh()
{
  __m256i a[NWORDS], r[NWORDS];
  ProPoint p;
  int i, seed;

  // initialize random generator
  seed = (int)time(NULL);
	srandom(seed);

  // randomize the input
  for (i = 0; i < NWORDS; i++) {
    a[i] = VSET64((uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random());
    r[i] = VSET64((uint32_t)random(), (uint32_t)random(), (uint32_t)random(), (uint32_t)random());
  }

  // benchmark part
  uint64_t start_cycles, end_cycles, diff_cycles;
  clock_t start_time, end_time;
  double tp;
  int iterations = 2000;

  // load cache
  for (i = 0; i < iterations; i++) keygen(r, r);
  // measure timing
  start_time = clock();
  start_cycles = read_tsc();
  for (i = 0; i < iterations; i++) {
    keygen(r, r);
    keygen(r, r);
    keygen(r, r);
    keygen(r, r);
    keygen(r, r);
    keygen(r, r);
    keygen(r, r);
    keygen(r, r);
    keygen(r, r);
    keygen(r, r);    
  }
  end_cycles = read_tsc();
  end_time = clock();
  diff_cycles = (end_cycles-start_cycles)/(10*iterations);
  puts("\n* Key Generation:");
  printf("  - Latency (4-Way ): %lld\n", diff_cycles);
  printf("  - Latency (single): %lld\n", diff_cycles/4);
  tp = 1e6*4*10*iterations / (double)(end_time-start_time);
  printf("  - Throughput: %8.1f op/sec\n", tp);


  // load cache
  for (i = 0; i < iterations; i++) sharedsecret(r, a, r);
  start_time = clock();
  start_cycles = read_tsc();
  for (i = 0; i < iterations; i++) { 
    sharedsecret(r, a, r);
    sharedsecret(r, a, r);
    sharedsecret(r, a, r);
    sharedsecret(r, a, r);
    sharedsecret(r, a, r);
    sharedsecret(r, a, r);
    sharedsecret(r, a, r);
    sharedsecret(r, a, r);
    sharedsecret(r, a, r);
    sharedsecret(r, a, r);
  }
  end_cycles = read_tsc();
  end_time = clock();
  diff_cycles = (end_cycles-start_cycles)/(10*iterations);
  puts("\n* Shared Secret:");
  printf("  - Latency (4-Way ): %lld\n", diff_cycles);
  printf("  - Latency (single): %lld\n", diff_cycles/4);
  tp = 1e6*4*10*iterations / (double)(end_time-start_time);
  printf("  - Throughput: %8.1f op/sec\n", tp);
}


/**
 * @brief Measure latency and throughput of all what we need.
 *
 * @details
 * Measure latency and throughput of field operations, point operations and 
 * scalar multiplications.
 */
void timing_all()
{
  puts("\n\n*******************************************************************");
  puts("TIMING OF SOFTWARE (clock cycles):");
  puts("-------------------------------------------------------------------");
  puts("Field operations:");
  timing_fp_arith();
  puts("-------------------------------------------------------------------");
  puts("Point operations:");
  timing_point_arith();
  puts("-------------------------------------------------------------------");
  puts("Diffie-Hellman functions:");
  timing_ecdh();
  puts("*******************************************************************");
}

int main()
{
  test_ecdh();
  timing_all();
  return 0;
}
