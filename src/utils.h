/**
 *******************************************************************************
 * @file utils.h
 * @version 1.0.1
 * @date 2020-09-01
 * @copyright Copyright © 2020 by University of Luxembourg.
 * @author Developed at SnT APSIA by: Hao Cheng, Johann Groszschaedl and Jiaqi Tian.
 *
 * @brief Header file of utils.
 *
 * @details 
 * This file contains utils such as conversion and print functions.
 *******************************************************************************
 */

#ifndef _UTILS_H
#define _UTILS_H

#include "gfparith.h"
#include <stdio.h>

// Return the clock cycle value of CPU
extern uint64_t read_tsc(void);

/**
 * @brief Conversion from mpi32 to mpi29.
 *
 * @details
 * Convert a multi-precision integer that is given as an array of 32-bit words
 * to an array of 29-bit words.
 *
 * @param r Mpi29 integer
 * @param a Mpi32 integer
 * @param rlen Length of r
 * @param alen Length of a
 */
void mpi29_conv_32to29(uint32_t *r, const uint32_t *a, int rlen, int alen)
{
  int i, j, shr_pos, shl_pos;
  uint32_t word, temp;

  i = j = 0;
  shr_pos=32; shl_pos=0;
  temp = 0;
  while ((i < rlen) && (j < alen)) {
    word = ((temp >> shr_pos) | (a[j] << shl_pos));
    r[i] = (word & MASK29);
    shr_pos-=3, shl_pos+=3;
    if ((shr_pos > 0) && (shl_pos < 32)) temp = a[j++];
    if (shr_pos <= 0) shr_pos += 32;
    if (shl_pos >= 32) shl_pos -= 32;
    // Any shift past 31 is undefined!
    if (shr_pos == 32) temp = 0;
    i++;
  }
  if (i < rlen) r[i++] = ((temp >> shr_pos) & MASK29);
  for (; i < rlen; i++) r[i] = 0;
}


/**
 * @brief Conversion from mpi29 to mpi32.
 *
 * @details
 * Convert a multi-precision integer that is given as an array of 29-bit words
 * to an array of 32-bit words. Note that 'rlen' can be smaller than 'alen'.
 *
 * @param r Mpi29 integer
 * @param a Mpi32 integer
 * @param rlen Length of r
 * @param alen Length of a
 */
void mpi29_conv_29to32(uint32_t *r, const uint32_t *a, int rlen, int alen)
{
  int i, j, bits_in_word, bits_to_shift;
  uint32_t word;

  i = j = 0;
  bits_in_word = bits_to_shift = 0;
  word = 0;
  while ((i < rlen) && (j < alen)) {
    word |= (a[j] << bits_in_word);
    bits_to_shift = (32 - bits_in_word);
    bits_in_word += 29;
    if (bits_in_word >= 32) {
      r[i++] = word;
      word = ((bits_to_shift > 0) ? (a[j] >> bits_to_shift) : 0);
      bits_in_word = ((bits_to_shift > 0) ? (29 - bits_to_shift) : 0);
    }
    j++;
  }
  if (i < rlen) r[i++] = word;
  for (; i < rlen; i++) r[i] = 0;
}

/**
 * @brief Print a multiprecision integer.
 *
 * @details
 * Print a multiprecision integer in an ordinary order with an string ahead.
 *
 * @param c Arbitry string
 * @param a Multiprecision integer
 * @param len Length of a
 */
void mpi29_print(const char *c, const uint32_t *a, int len)
{
  int i;

  printf("%s", c);
  for (i = len-1; i > 0; i--) printf("%08X", a[i]);
  printf("%08X\n", a[0]);
}

#endif
