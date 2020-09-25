/**
 *******************************************************************************
 * @file intrin.h
 * @version 1.0.1
 * @date 2020-09-01
 * @copyright Copyright © 2020 by University of Luxembourg.
 * @author Developed at SnT APSIA by: Hao Cheng, Johann Groszschaedl and Jiaqi Tian.
 *
 * @brief Header file of Intel AVX2 intrinsics. 
 *
 * @details 
 * Define some short names of AVX2 intrinsics for a clean code.
 *******************************************************************************
 */

#ifndef INTRIN_H
#define INTRIN_H

// AVX2 head file 
#include <immintrin.h>

// packed 64-bit arithmetics 
#define VADD(X, Y)         _mm256_add_epi64(X, Y)
#define VSUB(X, Y)         _mm256_sub_epi64(X, Y)
#define VMUL(X, Y)         _mm256_mul_epu32(X, Y)
#define VMAC(Z, X, Y)      VADD(Z, VMUL(X, Y))
#define VABS8(X)           _mm256_abs_epi8(X)
// bitwise logical operations
#define VXOR(X, Y)         _mm256_xor_si256(X, Y)
#define VAND(X, Y)         _mm256_and_si256(X, Y)
#define VOR(X, Y)          _mm256_or_si256(X, Y)
#define VSHR(X, Y)         _mm256_srli_epi64(X, Y)
#define VSHL(X, Y)         _mm256_slli_epi64(X, Y)
// the memory accessing and the broadcasting
#define VLOAD128(X)        _mm_load_si128((__m128i*)X)
#define VSET164(X)         _mm256_set1_epi64x(X)
#define VSET64(W, X, Y, Z) _mm256_set_epi64x(W, X, Y, Z)
#define VZERO              _mm256_setzero_si256()
#define VEXTR32(X, Y)      _mm256_extract_epi32(X, Y)
#define VSHUF32(X, Y)      _mm256_shuffle_epi32(X, Y)
#define VBROAD64(X)        _mm256_broadcastq_epi64(X)
#define VPERM64(X, Y)      _mm256_permute4x64_epi64(X, Y)


#endif
