/**
 *******************************************************************************
 * @file gfparith.c
 * @version 1.0.1
 * @date 2020-09-01
 * @copyright Copyright © 2020 by University of Luxembourg.
 * @author Developed at SnT APSIA by: Hao Cheng, Johann Groszschaedl and Jiaqi Tian.
 *
 * @brief C source file of field arithmetic.
 *
 * @details 
 * This file contains (4*1)-way vectorized field operations. 
 * The modulus p is 64*(2^255-19). 
 * Updates: Use local variables to represent all the limbs of the operands and 
 * results in each field operation function. From the experiments, this 
 * modification reduces the latency by 5% on Skylake. But increases the latency 
 * a little bit on Haswell.  
 *******************************************************************************
 */

#include "gfparith.h"


/**
 * @brief Conditional swap.
 *
 * @details
 * Replace (r,a) with (a,r) if b == 1;
 * replace (r,a) with (r,a) if b == 0.
 * Depending on a Boolean value that is passed as an argument to the function,
 * the two elements are either swapped or not swapped.
 *
 * @param r Field element
 * @param a Field element
 * @param b Swapping flag
 */
void mpi29_cswap_avx2(__m256i *r, __m256i *a, const __m256i b)
{
  __m256i r0 = r[0], r1 = r[1], r2 = r[2];
  __m256i r3 = r[3], r4 = r[4], r5 = r[5];
  __m256i r6 = r[6], r7 = r[7], r8 = r[8];
  __m256i a0 = a[0], a1 = a[1], a2 = a[2];
  __m256i a3 = a[3], a4 = a[4], a5 = a[5];
  __m256i a6 = a[6], a7 = a[7], a8 = a[8];
  __m256i x0, x1, x2, x3, x4, x5, x6, x7, x8;
  const __m256i mask = VSUB(VZERO, b);

  x0 = VXOR(r0, a0); x1 = VXOR(r1, a1); x2 = VXOR(r2, a2);
  x3 = VXOR(r3, a3); x4 = VXOR(r4, a4); x5 = VXOR(r5, a5);
  x6 = VXOR(r6, a6); x7 = VXOR(r7, a7); x8 = VXOR(r8, a8);

  x0 = VAND(x0, mask); x1 = VAND(x1, mask); x2 = VAND(x2, mask);
  x3 = VAND(x3, mask); x4 = VAND(x4, mask); x5 = VAND(x5, mask);
  x6 = VAND(x6, mask); x7 = VAND(x7, mask); x8 = VAND(x8, mask);

  r0 = VXOR(r0, x0); r1 = VXOR(r1, x1); r2 = VXOR(r2, x2);
  r3 = VXOR(r3, x3); r4 = VXOR(r4, x4); r5 = VXOR(r5, x5);
  r6 = VXOR(r6, x6); r7 = VXOR(r7, x7); r8 = VXOR(r8, x8);

  a0 = VXOR(a0, x0); a1 = VXOR(a1, x1); a2 = VXOR(a2, x2);
  a3 = VXOR(a3, x3); a4 = VXOR(a4, x4); a5 = VXOR(a5, x5);
  a6 = VXOR(a6, x6); a7 = VXOR(a7, x7); a8 = VXOR(a8, x8);

  r[0] = r0; r[1] = r1; r[2] = r2; 
  r[3] = r3; r[4] = r4; r[5] = r5;
  r[6] = r6; r[7] = r7; r[8] = r8;

  a[0] = a0; a[1] = a1; a[2] = a2;
  a[3] = a3; a[4] = a4; a[5] = a5;
  a[6] = a6; a[7] = a7; a[8] = a8;
}


/**
 * @brief Field addition.
 *
 * @details
 * r = a + b.
 * This is an ordinary addtion without reduction operation, which allows limbs
 * to expand one more bit.
 *
 * @param r Field element
 * @param a Field element
 * @param b Field element
 */
void mpi29_gfp_add_avx2(__m256i *r, const __m256i *a, const __m256i *b)
{
  __m256i a0 = a[0], a1 = a[1], a2 = a[2];
  __m256i a3 = a[3], a4 = a[4], a5 = a[5];
  __m256i a6 = a[6], a7 = a[7], a8 = a[8];
  __m256i b0 = b[0], b1 = b[1], b2 = b[2];
  __m256i b3 = b[3], b4 = b[4], b5 = b[5];
  __m256i b6 = b[6], b7 = b[7], b8 = b[8];
  __m256i r0, r1, r2, r3, r4, r5, r6, r7, r8;

  r0 = VADD(a0, b0); r1 = VADD(a1, b1); r2 = VADD(a2, b2);
  r3 = VADD(a3, b3); r4 = VADD(a4, b4); r5 = VADD(a5, b5);
  r6 = VADD(a6, b6); r7 = VADD(a7, b7); r8 = VADD(a8, b8);

  r[0] = r0; r[1] = r1; r[2] = r2; 
  r[3] = r3; r[4] = r4; r[5] = r5;
  r[6] = r6; r[7] = r7; r[8] = r8;
}


/**
 * @brief Field subtraction (no carry propagation and modular recution).
 *
 * @details
 * r = 2p + a - b.
 * This is an ordinary subtraction without carry propagation and modular reduction, 
 * which allows limbs to expand one more bit. It adds 2p to avoid any negative
 * intermediate values.
 * 
 * @param r Field element
 * @param a Field element
 * @param b Field element
 */
void mpi29_gfp_sub_avx2(__m256i *r, const __m256i *a, const __m256i *b)
{
  __m256i a0 = a[0], a1 = a[1], a2 = a[2];
  __m256i a3 = a[3], a4 = a[4], a5 = a[5];
  __m256i a6 = a[6], a7 = a[7], a8 = a[8];
  __m256i b0 = b[0], b1 = b[1], b2 = b[2];
  __m256i b3 = b[3], b4 = b[4], b5 = b[5];
  __m256i b6 = b[6], b7 = b[7], b8 = b[8];
  __m256i r0, r1, r2, r3, r4, r5, r6, r7, r8;
  const __m256i VDLSWP = VSET164(LSWP29*2);
  const __m256i VDWRDP = VSET164(MASK29*2);

  // NOTE: if put (a[i]-b[i]) at the latter position, it is faster.
  r0 = VADD(VDLSWP, VSUB(a0, b0));
  r1 = VADD(VDWRDP, VSUB(a1, b1));
  r2 = VADD(VDWRDP, VSUB(a2, b2));
  r3 = VADD(VDWRDP, VSUB(a3, b3));
  r4 = VADD(VDWRDP, VSUB(a4, b4));
  r5 = VADD(VDWRDP, VSUB(a5, b5));
  r6 = VADD(VDWRDP, VSUB(a6, b6));
  r7 = VADD(VDWRDP, VSUB(a7, b7));
  r8 = VADD(VDWRDP, VSUB(a8, b8));

  r[0] = r0; r[1] = r1; r[2] = r2; 
  r[3] = r3; r[4] = r4; r[5] = r5;
  r[6] = r6; r[7] = r7; r[8] = r8;
}


/**
 * @brief Field subtraction (including carry propagation and modular reduction).
 *
 * @details
 * r = 2p + a - b mod p.
 * This is a modular subtraction. It adds 2p to avoid any negative intermediate 
 * values.
 *
 * @param r Field element
 * @param a Field element
 * @param b Field element
 */
void mpi29_gfp_sbc_avx2(__m256i *r, const __m256i *a, const __m256i *b)
{
  __m256i a0 = a[0], a1 = a[1], a2 = a[2];
  __m256i a3 = a[3], a4 = a[4], a5 = a[5];
  __m256i a6 = a[6], a7 = a[7], a8 = a[8];
  __m256i b0 = b[0], b1 = b[1], b2 = b[2];
  __m256i b3 = b[3], b4 = b[4], b5 = b[5];
  __m256i b6 = b[6], b7 = b[7], b8 = b[8];
  __m256i r0, r1, r2, r3, r4, r5, r6, r7, r8, temp;
  const __m256i VDLSWP  = VSET164(LSWP29*2);
  const __m256i VDWRDP  = VSET164(MASK29*2);
  const __m256i VMASK29 = VSET164(MASK29);
  const __m256i VCONSTC = VSET164(CONSTC);

  // subtraction loop
  // NOTE: if put (a[i]-b[i]) at the latter position, it is faster.
  r0 = VADD(VDLSWP, VSUB(a0, b0));
  r1 = VADD(VDWRDP, VSUB(a1, b1));
  r2 = VADD(VDWRDP, VSUB(a2, b2));
  r3 = VADD(VDWRDP, VSUB(a3, b3));
  r4 = VADD(VDWRDP, VSUB(a4, b4));
  r5 = VADD(VDWRDP, VSUB(a5, b5));
  r6 = VADD(VDWRDP, VSUB(a6, b6));
  r7 = VADD(VDWRDP, VSUB(a7, b7));
  r8 = VADD(VDWRDP, VSUB(a8, b8));

  // carry propagation loop
  r1 = VADD(r1, VSHR(r0, BITS29)); r0 = VAND(r0, VMASK29);
  r2 = VADD(r2, VSHR(r1, BITS29)); r1 = VAND(r1, VMASK29);
  r3 = VADD(r3, VSHR(r2, BITS29)); r2 = VAND(r2, VMASK29);
  r4 = VADD(r4, VSHR(r3, BITS29)); r3 = VAND(r3, VMASK29);
  r5 = VADD(r5, VSHR(r4, BITS29)); r4 = VAND(r4, VMASK29);
  r6 = VADD(r6, VSHR(r5, BITS29)); r5 = VAND(r5, VMASK29);
  r7 = VADD(r7, VSHR(r6, BITS29)); r6 = VAND(r6, VMASK29);
  r8 = VADD(r8, VSHR(r7, BITS29)); r7 = VAND(r7, VMASK29);

  // the final step to compute r0 and r8
  temp = VSHR(r8, BITS29); 
  temp = VMUL(temp, VCONSTC);
  r0   = VADD(r0, temp); 
  r8   = VAND(r8, VMASK29);

  r[0] = r0; r[1] = r1; r[2] = r2; 
  r[3] = r3; r[4] = r4; r[5] = r5;
  r[6] = r6; r[7] = r7; r[8] = r8;
}


/**
 * @brief Field multiplication.
 *
 * @details
 * r = a * b mod p.
 * This is a modular multiplication. It performs a product-scanning and modulo-p 
 * reduction separately. It uses local variables to store intermediate values.  
 * 
 * @param r Field element
 * @param a Field element
 * @param b Field element
 */
void mpi29_gfp_mul_avx2(__m256i *r, const __m256i *a, const __m256i *b)
{
  __m256i a0 = a[0], a1 = a[1], a2 = a[2];
  __m256i a3 = a[3], a4 = a[4], a5 = a[5];
  __m256i a6 = a[6], a7 = a[7], a8 = a[8];
  __m256i b0 = b[0], b1 = b[1], b2 = b[2];
  __m256i b3 = b[3], b4 = b[4], b5 = b[5];
  __m256i b6 = b[6], b7 = b[7], b8 = b[8];
  __m256i r0, r1, r2, r3, r4, r5, r6, r7, r8;
  __m256i t0, t1, t2, t3, t4, t5, t6, t7, t8, accu;
  const __m256i VMASK29 = VSET164(MASK29);
  const __m256i VCONSTC = VSET164(CONSTC);

  // 1st loop of the product-scanning multiplication
  t0 = VMUL(    a0, b0);

  t1 = VMUL(    a0, b1); t1 = VMAC(t1, a1, b0);

  t2 = VMUL(    a0, b2); t2 = VMAC(t2, a1, b1); t2 = VMAC(t2, a2, b0);

  t3 = VMUL(    a0, b3); t3 = VMAC(t3, a1, b2); t3 = VMAC(t3, a2, b1);
  t3 = VMAC(t3, a3, b0);

  t4 = VMUL(    a0, b4); t4 = VMAC(t4, a1, b3); t4 = VMAC(t4, a2, b2);
  t4 = VMAC(t4, a3, b1); t4 = VMAC(t4, a4, b0);

  t5 = VMUL(    a0, b5); t5 = VMAC(t5, a1, b4); t5 = VMAC(t5, a2, b3);
  t5 = VMAC(t5, a3, b2); t5 = VMAC(t5, a4, b1); t5 = VMAC(t5, a5, b0);

  t6 = VMUL(    a0, b6); t6 = VMAC(t6, a1, b5); t6 = VMAC(t6, a2, b4);
  t6 = VMAC(t6, a3, b3); t6 = VMAC(t6, a4, b2); t6 = VMAC(t6, a5, b1);
  t6 = VMAC(t6, a6, b0);

  t7 = VMUL(    a0, b7); t7 = VMAC(t7, a1, b6); t7 = VMAC(t7, a2, b5);
  t7 = VMAC(t7, a3, b4); t7 = VMAC(t7, a4, b3); t7 = VMAC(t7, a5, b2);
  t7 = VMAC(t7, a6, b1); t7 = VMAC(t7, a7, b0);

  t8 = VMUL(    a0, b8); t8 = VMAC(t8, a1, b7); t8 = VMAC(t8, a2, b6);
  t8 = VMAC(t8, a3, b5); t8 = VMAC(t8, a4, b4); t8 = VMAC(t8, a5, b3);
  t8 = VMAC(t8, a6, b2); t8 = VMAC(t8, a7, b1); t8 = VMAC(t8, a8, b0);

  accu = VSHR(t8, BITS29);
  t8   = VAND(t8, VMASK29);

  // 2nd loop of the product-scanning multiplication
  accu = VMAC(accu, a1, b8); accu = VMAC(accu, a2, b7);
  accu = VMAC(accu, a3, b6); accu = VMAC(accu, a4, b5);
  accu = VMAC(accu, a5, b4); accu = VMAC(accu, a6, b3);
  accu = VMAC(accu, a7, b2); accu = VMAC(accu, a8, b1);
  r0   = VAND(accu, VMASK29); 
  accu = VSHR(accu, BITS29);

  accu = VMAC(accu, a2, b8); accu = VMAC(accu, a3, b7);
  accu = VMAC(accu, a4, b6); accu = VMAC(accu, a5, b5);
  accu = VMAC(accu, a6, b4); accu = VMAC(accu, a7, b3);
  accu = VMAC(accu, a8, b2);
  r1   = VAND(accu, VMASK29);
  accu = VSHR(accu, BITS29);

  accu = VMAC(accu, a3, b8); accu = VMAC(accu, a4, b7);
  accu = VMAC(accu, a5, b6); accu = VMAC(accu, a6, b5);
  accu = VMAC(accu, a7, b4); accu = VMAC(accu, a8, b3);
  r2   = VAND(accu, VMASK29);
  accu = VSHR(accu, BITS29);

  accu = VMAC(accu, a4, b8); accu = VMAC(accu, a5, b7);
  accu = VMAC(accu, a6, b6); accu = VMAC(accu, a7, b5);
  accu = VMAC(accu, a8, b4);
  r3   = VAND(accu, VMASK29);
  accu = VSHR(accu, BITS29);

  accu = VMAC(accu, a5, b8); accu = VMAC(accu, a6, b7);
  accu = VMAC(accu, a7, b6); accu = VMAC(accu, a8, b5);
  r4   = VAND(accu, VMASK29);
  accu = VSHR(accu, BITS29);

  accu = VMAC(accu, a6, b8); accu = VMAC(accu, a7, b7);
  accu = VMAC(accu, a8, b6);
  r5   = VAND(accu, VMASK29);
  accu = VSHR(accu, BITS29);

  accu = VMAC(accu, a7, b8); accu = VMAC(accu, a8, b7);
  r6   = VAND(accu, VMASK29);
  accu = VSHR(accu, BITS29);

  accu = VMAC(accu, a8, b8);
  r7   = VAND(accu, VMASK29);

  r8   = VSHR(accu, BITS29);

  // modulo-p reduction and conversion to 29-bit limbs
  accu = VMAC(t0, r0, VCONSTC);
  r0   = VAND(accu, VMASK29);

  accu = VADD(t1, VSHR(accu, BITS29)); accu = VMAC(accu, r1, VCONSTC);
  r1   = VAND(accu, VMASK29);

  accu = VADD(t2, VSHR(accu, BITS29)); accu = VMAC(accu, r2, VCONSTC);
  r2   = VAND(accu, VMASK29);

  accu = VADD(t3, VSHR(accu, BITS29)); accu = VMAC(accu, r3, VCONSTC);
  r3   = VAND(accu, VMASK29);

  accu = VADD(t4, VSHR(accu, BITS29)); accu = VMAC(accu, r4, VCONSTC);
  r4   = VAND(accu, VMASK29);

  accu = VADD(t5, VSHR(accu, BITS29)); accu = VMAC(accu, r5, VCONSTC);
  r5   = VAND(accu, VMASK29);

  accu = VADD(t6, VSHR(accu, BITS29)); accu = VMAC(accu, r6, VCONSTC);
  r6   = VAND(accu, VMASK29);

  accu = VADD(t7, VSHR(accu, BITS29)); accu = VMAC(accu, r7, VCONSTC);
  r7   = VAND(accu, VMASK29);

  accu = VADD(t8, VSHR(accu, BITS29)); accu = VMAC(accu, r8, VCONSTC);
  r8   = VAND(accu, VMASK29);

  accu = VSHR(accu, BITS29);
  r0   = VMAC(r0, accu, VCONSTC);

  r[0] = r0; r[1] = r1; r[2] = r2; 
  r[3] = r3; r[4] = r4; r[5] = r5;
  r[6] = r6; r[7] = r7; r[8] = r8;
}


/**
 * @brief Field scalar multiplication.
 *
 * @details
 * r = b * a mod p.
 * The modular multiplication between a field element "a" and a 29-bit integer "b". 
 * 
 * @param r Field element
 * @param a Field element 
 * @param b 29-bit integer
 */
void mpi29_gfp_mul29_avx2(__m256i *r, const __m256i *a, const uint32_t b)
{
  __m256i a0 = a[0], a1 = a[1], a2 = a[2];
  __m256i a3 = a[3], a4 = a[4], a5 = a[5];
  __m256i a6 = a[6], a7 = a[7], a8 = a[8];
  __m256i r0, r1, r2, r3, r4, r5, r6, r7, r8, accu;
  const __m256i vb = VSET164(b);
  const __m256i VMASK29 = VSET164(MASK29);
  const __m256i VCONSTC = VSET164(CONSTC);

  accu = VMUL(a0, vb); r0 = VAND(accu, VMASK29); accu = VSHR(accu, BITS29);
  accu = VMAC(accu, a1, vb); r1 = VAND(accu, VMASK29); accu = VSHR(accu, BITS29);
  accu = VMAC(accu, a2, vb); r2 = VAND(accu, VMASK29); accu = VSHR(accu, BITS29);
  accu = VMAC(accu, a3, vb); r3 = VAND(accu, VMASK29); accu = VSHR(accu, BITS29);
  accu = VMAC(accu, a4, vb); r4 = VAND(accu, VMASK29); accu = VSHR(accu, BITS29);
  accu = VMAC(accu, a5, vb); r5 = VAND(accu, VMASK29); accu = VSHR(accu, BITS29);
  accu = VMAC(accu, a6, vb); r6 = VAND(accu, VMASK29); accu = VSHR(accu, BITS29);
  accu = VMAC(accu, a7, vb); r7 = VAND(accu, VMASK29); accu = VSHR(accu, BITS29);
  accu = VMAC(accu, a8, vb); r8 = VAND(accu, VMASK29); 

  accu = VMUL(VCONSTC, VSHR(accu, BITS29));
  r0   = VADD(r0, VAND(accu, VMASK29));
  r1   = VADD(r1, VSHR(accu, BITS29));

  r[0] = r0; r[1] = r1; r[2] = r2; 
  r[3] = r3; r[4] = r4; r[5] = r5;
  r[6] = r6; r[7] = r7; r[8] = r8;
}


/**
 * @brief Field squaring.
 *
 * @details
 * r = a^2 mod p.
 * This is a modular squaring. It performs a product-scanning and modulo-p 
 * reduction separately. It uses local variables to store intermediate values.  
 * 
 * @param r Field element
 * @param a Field element
 */
void mpi29_gfp_sqr_avx2(__m256i *r, const __m256i *a)
{
  __m256i a0 = a[0], a1 = a[1], a2 = a[2];
  __m256i a3 = a[3], a4 = a[4], a5 = a[5];
  __m256i a6 = a[6], a7 = a[7], a8 = a[8];
  __m256i r0, r1, r2, r3, r4, r5, r6, r7, r8;
  __m256i t0, t1, t2, t3, t4, t5, t6, t7, t8, accu, temp;
  const __m256i VMASK29 = VSET164(MASK29);
  const __m256i VCONSTC = VSET164(CONSTC);

  // 1st loop of the product-scanning squaring
  t0   = VMUL(a0, a0);

  accu = VMUL(a0, a1); 
  t1   = VSHL(accu, 1);

  accu = VMUL(a0, a2); 
  t2   = VSHL(accu, 1); t2 = VMAC(t2, a1, a1);

  accu = VMUL(a0, a3); 
  accu = VMAC(accu, a1, a2); 
  t3   = VSHL(accu, 1);

  accu = VMUL(a0, a4); accu = VMAC(accu, a1, a3);
  t4   = VSHL(accu, 1); t4  = VMAC(t4, a2, a2);

  accu = VMUL(a0, a5); accu = VMAC(accu, a1, a4); accu = VMAC(accu, a2, a3);
  t5   = VSHL(accu, 1);

  accu = VMUL(a0, a6); accu = VMAC(accu, a1, a5); accu = VMAC(accu, a2, a4);
  t6   = VSHL(accu, 1); t6  = VMAC(t6, a3, a3);

  accu = VMUL(a0, a7); accu = VMAC(accu, a1, a6); accu = VMAC(accu, a2, a5);
  accu = VMAC(accu, a3, a4);
  t7   = VSHL(accu, 1);

  accu = VMUL(a0, a8); accu = VMAC(accu, a1, a7); accu = VMAC(accu, a2, a6);
  accu = VMAC(accu, a3, a5);
  t8   = VSHL(accu, 1); t8  = VMAC(t8, a4, a4);

  temp = VSHR(t8, BITS29); t8 = VAND(t8, VMASK29);

  // 2nd loop of the product-scanning squaring
  accu = VMUL(a1, a8); accu = VMAC(accu, a2, a7); accu = VMAC(accu, a3, a6);
  accu = VMAC(accu, a4, a5);
  temp = VADD(temp, VSHL(accu, 1));
  r0   = VAND(temp, VMASK29);
  temp = VSHR(temp, BITS29);

  accu = VMUL(a2, a8); accu = VMAC(accu, a3, a7); accu = VMAC(accu, a4, a6);
  temp = VADD(temp, VSHL(accu, 1));
  temp = VMAC(temp, a5, a5);
  r1   = VAND(temp, VMASK29);
  temp = VSHR(temp, BITS29);

  accu = VMUL(a3, a8); accu = VMAC(accu, a4, a7); accu = VMAC(accu, a5, a6);
  temp = VADD(temp, VSHL(accu, 1));
  r2   = VAND(temp, VMASK29);
  temp = VSHR(temp, BITS29);

  accu = VMUL(a4, a8); accu = VMAC(accu, a5, a7);
  temp = VADD(temp, VSHL(accu, 1));
  temp = VMAC(temp, a6, a6);
  r3   = VAND(temp, VMASK29);
  temp = VSHR(temp, BITS29);

  accu = VMUL(a5, a8); accu = VMAC(accu, a6, a7);
  temp = VADD(temp, VSHL(accu, 1));
  r4   = VAND(temp, VMASK29);
  temp = VSHR(temp, BITS29);

  accu = VMUL(a6, a8);
  temp = VADD(temp, VSHL(accu, 1));
  temp = VMAC(temp, a7, a7);
  r5   = VAND(temp, VMASK29);
  temp = VSHR(temp, BITS29);

  accu = VMUL(a7, a8);
  temp = VADD(temp, VSHL(accu, 1));
  r6   = VAND(temp, VMASK29);
  temp = VSHR(temp, BITS29);

  temp = VMAC(temp, a8, a8);
  r7   = VAND(temp, VMASK29);
  r8   = VSHR(temp, BITS29);

  // modulo reduction and conversion to 29-bit limbs
  accu = VADD(t0, VMUL(r0, VCONSTC));
  r0   = VAND(accu, VMASK29);

  accu = VADD(t1, VSHR(accu, BITS29)); accu = VMAC(accu, r1, VCONSTC);
  r1   = VAND(accu, VMASK29);

  accu = VADD(t2, VSHR(accu, BITS29)); accu = VMAC(accu, r2, VCONSTC);
  r2   = VAND(accu, VMASK29);

  accu = VADD(t3, VSHR(accu, BITS29)); accu = VMAC(accu, r3, VCONSTC);
  r3   = VAND(accu, VMASK29);

  accu = VADD(t4, VSHR(accu, BITS29)); accu = VMAC(accu, r4, VCONSTC);
  r4   = VAND(accu, VMASK29);

  accu = VADD(t5, VSHR(accu, BITS29)); accu = VMAC(accu, r5, VCONSTC);
  r5   = VAND(accu, VMASK29);

  accu = VADD(t6, VSHR(accu, BITS29)); accu = VMAC(accu, r6, VCONSTC);
  r6   = VAND(accu, VMASK29);

  accu = VADD(t7, VSHR(accu, BITS29)); accu = VMAC(accu, r7, VCONSTC);
  r7   = VAND(accu, VMASK29);

  accu = VADD(t8, VSHR(accu, BITS29)); accu = VMAC(accu, r8, VCONSTC);
  r8   = VAND(accu, VMASK29);

  accu = VSHR(accu, BITS29);
  r0   = VADD(r0, VMUL(accu, VCONSTC));

  r[0] = r0; r[1] = r1; r[2] = r2; 
  r[3] = r3; r[4] = r4; r[5] = r5;
  r[6] = r6; r[7] = r7; r[8] = r8;
}


/**
 * @brief Field multiplicative inversion.
 *
 * @details
 * r = a^-1 mod p.
 * This function computes the multiplicative inverse of an element.
 * 
 * @param r Field element
 * @param a Field element
 */
void mpi29_gfp_inv_avx2(__m256i *r, const __m256i *a)
{
  __m256i t0[NWORDS], t1[NWORDS], t2[NWORDS], t3[NWORDS];
  int i;

  mpi29_gfp_sqr_avx2(t0, a);
  mpi29_gfp_sqr_avx2(t1, t0);
  mpi29_gfp_sqr_avx2(t1, t1);
  mpi29_gfp_mul_avx2(t1, a, t1);
  mpi29_gfp_mul_avx2(t0, t0, t1);
  mpi29_gfp_sqr_avx2(t2, t0);
  mpi29_gfp_mul_avx2(t1, t1, t2);
  mpi29_gfp_sqr_avx2(t2, t1);
  for (i = 0; i < 4; i++) mpi29_gfp_sqr_avx2(t2, t2);
  mpi29_gfp_mul_avx2(t1, t2, t1);
  mpi29_gfp_sqr_avx2(t2, t1);
  for (i = 0; i < 9; i++) mpi29_gfp_sqr_avx2(t2, t2);
  mpi29_gfp_mul_avx2(t2, t2, t1);
  mpi29_gfp_sqr_avx2(t3, t2);
  for (i = 0; i < 19; i++) mpi29_gfp_sqr_avx2(t3, t3);   
  mpi29_gfp_mul_avx2(t2, t3, t2);
  mpi29_gfp_sqr_avx2(t2, t2);
  for (i = 0; i < 9; i++) mpi29_gfp_sqr_avx2(t2, t2);
  mpi29_gfp_mul_avx2(t1, t2, t1);
  mpi29_gfp_sqr_avx2(t2, t1);
  for (i = 0; i < 49; i++) mpi29_gfp_sqr_avx2(t2, t2);
  mpi29_gfp_mul_avx2(t2, t2, t1);
  mpi29_gfp_sqr_avx2(t3, t2);
  for (i = 0; i < 99; i++) mpi29_gfp_sqr_avx2(t3, t3);
  mpi29_gfp_mul_avx2(t2, t3, t2);
  mpi29_gfp_sqr_avx2(t2, t2);
  for (i = 0; i < 49; i++) mpi29_gfp_sqr_avx2(t2, t2);
  mpi29_gfp_mul_avx2(t1, t2, t1);
  mpi29_gfp_sqr_avx2(t1, t1);
  for (i = 0; i < 4; i++) mpi29_gfp_sqr_avx2(t1, t1);
  mpi29_gfp_mul_avx2(r, t1, t0);
}

/**
 * @brief Copy.
 *
 * @details
 * Copy a to r.
 *
 * @param r Field element
 * @param a Field element
 */
void mpi29_copy_avx2(__m256i *r, const __m256i *a)
{
  r[0] = a[0];
  r[1] = a[1];
  r[2] = a[2];
  r[3] = a[3];
  r[4] = a[4];
  r[5] = a[5];
  r[6] = a[6];
  r[7] = a[7];
  r[8] = a[8];
}
