/**
 *******************************************************************************
 * @file ecdh.c
 * @version 1.0.1
 * @date 2020-09-01
 * @copyright Copyright © 2020 by University of Luxembourg.
 * @author Developed at SnT APSIA by: Hao Cheng, Johann Groszschaedl and Jiaqi Tian.
 *
 * @brief C source file of Diffie-Hellman key exchange functions
 *
 * @details 
 * This file contains key generation and the computation of shared secret.
 *******************************************************************************
 */

#include "moncurve.h"
#include "tedcurve.h"

/**
 * @brief The final step to reduce pk or ss by modulo p 
 *
 * @details
 * Perform modulo-p reduction for the integer to make it in [0, 2^255-19)
 * 
 * @param r Field element
 * @param a Field element
 */
static void final_modp (__m256i *a)
{
  __m256i a0 = a[0], a1 = a[1], a2 = a[2];
  __m256i a3 = a[3], a4 = a[4], a5 = a[5];
  __m256i a6 = a[6], a7 = a[7], a8 = a[8];
  __m256i temp;
  const __m256i VMASK23 = VSET164(0x7FFFFFUL);
  const __m256i VMASK29 = VSET164(MASK29);
  const __m256i V19 = VSET164(19);

  // current r is 9*29-bit, we will convert it to 8*29-bit + 23-bit
  temp = VSHR(a8, 23); a8 = VAND(a8, VMASK23);
  // carry propagation
  a0 = VADD(a0, VMUL(temp, V19));  
  a1 = VADD(a1, VSHR(a0, BITS29)); a0 = VAND(a0, VMASK29);
  a2 = VADD(a2, VSHR(a1, BITS29)); a1 = VAND(a1, VMASK29);
  a3 = VADD(a3, VSHR(a2, BITS29)); a2 = VAND(a2, VMASK29);
  a4 = VADD(a4, VSHR(a3, BITS29)); a3 = VAND(a3, VMASK29);
  a5 = VADD(a5, VSHR(a4, BITS29)); a4 = VAND(a4, VMASK29);
  a6 = VADD(a6, VSHR(a5, BITS29)); a5 = VAND(a5, VMASK29);
  a7 = VADD(a7, VSHR(a6, BITS29)); a6 = VAND(a6, VMASK29);
  a8 = VADD(a8, VSHR(a7, BITS29)); a7 = VAND(a7, VMASK29);

  // it is possible that r8 is still longer than 23-bit, repeat the above operations
  temp = VSHR(a8, 23); a8 = VAND(a8, VMASK23);
  // carry propagation
  a0 = VADD(a0, VMUL(temp, V19));  
  a1 = VADD(a1, VSHR(a0, BITS29)); a0 = VAND(a0, VMASK29);
  a2 = VADD(a2, VSHR(a1, BITS29)); a1 = VAND(a1, VMASK29);
  a3 = VADD(a3, VSHR(a2, BITS29)); a2 = VAND(a2, VMASK29);
  a4 = VADD(a4, VSHR(a3, BITS29)); a3 = VAND(a3, VMASK29);
  a5 = VADD(a5, VSHR(a4, BITS29)); a4 = VAND(a4, VMASK29);
  a6 = VADD(a6, VSHR(a5, BITS29)); a5 = VAND(a5, VMASK29);
  a7 = VADD(a7, VSHR(a6, BITS29)); a6 = VAND(a6, VMASK29);
  a8 = VADD(a8, VSHR(a7, BITS29)); a7 = VAND(a7, VMASK29);

  a[0] = a0; a[1] = a1; a[2] = a2; 
  a[3] = a3; a[4] = a4; a[5] = a5;
  a[6] = a6; a[7] = a7; a[8] = a8;
}


/**
 * @brief Key generation.
 *
 * @details
 * Generate public key based on the given private key. 
 * 
 * @param pk Public key
 * @param sk Private key
 */
void keygen(__m256i *pk, const __m256i *sk)
{
  mon_mul_fixbase_avx2(pk, sk);
  final_modp(pk);
}

/**
 * @brief Shared secret computation.
 *
 * @details
 * Generate a shared secret (session key) based on own private key and the public 
 * key of the other side.
 * 
 * @param ss  Shared secret
 * @param ska Own private key
 * @param pkb Public key of the other side
 */
void sharedsecret(__m256i *ss, const __m256i *ska, const __m256i *pkb)
{
  // Variable-base point scalar multiplication on Montgomery curve
  mon_mul_varbase_avx2(ss, ska, pkb);
  final_modp(ss);
}
