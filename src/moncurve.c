/**
 *******************************************************************************
 * @file moncurve.c
 * @version 1.0.1
 * @date 2020-09-01
 * @copyright Copyright © 2020 by University of Luxembourg.
 * @author Developed at SnT APSIA by: Hao Cheng, Johann Groszschaedl and Jiaqi Tian.
 *
 * @brief C source file of point arithmetic on Montgomery curve.
 *
 * @details 
 * This file contains (4*1)-way parallel point operations on Montgomery curve. 
 *******************************************************************************
 */

#include "moncurve.h"
#include "tedcurve.h"


/**
 * @brief Montgomery ladder step.
 *
 * @details
 * (P,Q) <- LadderStep(P,Q,pk)
 * The Ladder-step contains a differential point addition and point doubling, 
 * and it only operates on x- and z-coordinates of projective points.  
 * 
 * @param p Projective point 
 * @param q Projective point 
 * @param xd Field element
 */
void mon_ladder_step_avx2(ProPoint *p, ProPoint *q, const __m256i *xd)
{
  // we use y-coordinate as tmp1, tmp2
  __m256i *tmp1 = p->y, *tmp2 = q->y;

  mpi29_gfp_add_avx2(tmp1, p->x, p->z);
  mpi29_gfp_sbc_avx2(p->x, p->x, p->z);
  mpi29_gfp_add_avx2(tmp2, q->x, q->z);
  mpi29_gfp_sub_avx2(q->x, q->x, q->z);
  mpi29_gfp_sqr_avx2(p->z, tmp1);
  mpi29_gfp_mul_avx2(q->z, tmp2, p->x);
  mpi29_gfp_mul_avx2(tmp2, q->x, tmp1);
  mpi29_gfp_sqr_avx2(tmp1, p->x);
  mpi29_gfp_mul_avx2(p->x, p->z, tmp1);
  mpi29_gfp_sub_avx2(tmp1, p->z, tmp1);
  mpi29_gfp_mul29_avx2(q->x, tmp1, (CONSTA-2)/4);
  mpi29_gfp_add_avx2(q->x, q->x, p->z);
  mpi29_gfp_mul_avx2(p->z, q->x, tmp1);
  mpi29_gfp_add_avx2(tmp1, tmp2, q->z);
  mpi29_gfp_sqr_avx2(q->x, tmp1);
  mpi29_gfp_sbc_avx2(tmp1, tmp2, q->z);
  mpi29_gfp_sqr_avx2(tmp2, tmp1);
  mpi29_gfp_mul_avx2(q->z, tmp2, xd);
}


/**
 * @brief Conditional swap (cswap) of two points.
 *
 * @details
 * Replace (P,Q) with (Q,P) if b == 1;
 * replace (P,Q) with (P,Q) if b == 0.
 * Depending on a boolean value that is passed as argument to the function,
 * the two points are either swapped or not swapped. 
 * 
 * @param p Projective point
 * @param q Projective point
 * @param b Swapping flag
 */
static void mon_cswap_point_avx2(ProPoint *p, ProPoint *q, const __m256i b)
{
  const __m256i one = VSET164(1);
  const __m256i cbit = VAND(b, one);

  mpi29_cswap_avx2(p->x, q->x, cbit);
  mpi29_cswap_avx2(p->z, q->z, cbit);
}


/**
 * @brief Variable-base scalar multiplication.
 *
 * @details
 * xR = k * xP.
 * This function computes only the x-coordinate of R = k * P, where R and P are
 * points with affine coordinates. This is the core operation of ECDH shared 
 * secret phase. 
 * 
 * @param r x-coordinate of point with affine coordinates
 * @param k scalar 
 * @param x x-coordinate of point with affine coordinates
 */
void mon_mul_varbase_avx2(__m256i *r, const __m256i *k, const __m256i *x)
{
  ProPoint p1, p2;
  __m256i b, s = VZERO, kp[8];
  const __m256i t0 = VSET164(0xFFFFFFF8UL);
  const __m256i t1 = VSET164(0x7FFFFFFFUL);
  const __m256i t2 = VSET164(0x40000000UL);
  int i;

  // prune scalar k
  for (i = 0; i < 8; i++) kp[i] = k[i];
  kp[0] = VAND(kp[0], t0);
  kp[7] = VAND(kp[7], t1);
  kp[7] = VOR(kp[7], t2);

  // initialize ladder
  for (i = 0; i < NWORDS; i++) {
    p1.x[i] = p1.z[i] = p2.z[i] = VZERO;
    p2.x[i] = x[i];
  }
  p1.x[0] = p2.z[0] = VSET164(1);

  // main ladder loop
  for (i = 254; i >= 0; i--) {
  b = kp[i>>5];
  b = VSHR(b, i&31);
  s = VXOR(s, b);
  mon_cswap_point_avx2(&p1, &p2, s);
  mon_ladder_step_avx2(&p1, &p2, x);
  s = b;
}
  mon_cswap_point_avx2(&p1, &p2, s);

  // projective -> affine
  mpi29_gfp_inv_avx2(p2.y, p1.z);
  mpi29_gfp_mul_avx2(r, p2.y, p1.x);
}


/**
 * @brief Fixed-base scalar multiplication on Montgomery curve.
 *
 * @details
 * R = k * B.
 * Take advantage of fixed-base scalar multiplication on twsited Edwards curve 
 * and then map the projective points to Montgomery curve. Finally output the 
 * x-coordinate of R on Montgomery curve.
 * 
 * @param r Projective point 
 * @param k Scalar 
 */
void mon_mul_fixbase_avx2(__m256i *r, const __m256i *k)
{
  ProPoint p;
  __m256i t[NWORDS];

  ted_mul_fixbase_avx2(&p, k);
  // from twisted Edwards curve to Montgomery curve u = (z+y)/(z-y)
  mpi29_gfp_sbc_avx2(p.x, p.z, p.y);   // t1 = z-y
  mpi29_gfp_inv_avx2(p.x, p.x);        // t1 = 1/(z-y) 
  mpi29_gfp_add_avx2(t, p.z, p.y);     // t2 = z+y
  mpi29_gfp_mul_avx2(r, t, p.x);       // r = (z+y)/(z-y)
}
