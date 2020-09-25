/**
 *******************************************************************************
 * @file gfparith.h
 * @version 1.0.1
 * @date 2020-09-01
 * @copyright Copyright © 2020 by University of Luxembourg.
 * @author Developed at SnT APSIA by: Hao Cheng, Johann Groszschaedl and Jiaqi Tian.
 *
 * @brief Header file of field arithmetic.
 *
 * @details 
 * This file contains some constants and function prototypes of filed arithmetic. 
 *******************************************************************************
 */

#ifndef _GFPARITH_H
#define _GFPARITH_H

#include "intrin.h"
#include <stdint.h>

// we use a radix-2^29 for the field elements
#define NWORDS 9
#define BITS29 29
#define MASK29 0x1FFFFFFFUL
#define CONSTC 1216
#define CONSTA 486662
// least significant 29-bit word of p = 64*(2^255 - 19) = 2^261 - 1216
#define LSWP29 0x1FFFFB40UL

// function prototypes

void mpi29_gfp_add_avx2(__m256i *r, const __m256i *a, const __m256i *b);
void mpi29_gfp_sub_avx2(__m256i *r, const __m256i *a, const __m256i *b);
void mpi29_gfp_sbc_avx2(__m256i *r, const __m256i *a, const __m256i *b);
void mpi29_gfp_mul_avx2(__m256i *r, const __m256i *a, const __m256i *b);
void mpi29_gfp_mul29_avx2(__m256i *r, const __m256i *a, const uint32_t b);
void mpi29_gfp_sqr_avx2(__m256i *r, const __m256i *a);
void mpi29_gfp_inv_avx2(__m256i *r, const __m256i *a);
void mpi29_cswap_avx2(__m256i *r, __m256i *a, const __m256i b);
void mpi29_copy_avx2(__m256i *r, const __m256i *a);
#endif
