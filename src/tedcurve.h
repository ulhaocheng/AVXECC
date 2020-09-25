/**
 *******************************************************************************
 * @file tedcurve.h
 * @version 1.0.1
 * @date 2020-09-01
 * @copyright Copyright © 2020 by University of Luxembourg.
 * @author Developed at SnT APSIA by: Hao Cheng, Johann Groszschaedl and Jiaqi Tian.
 *
 * @brief Header file of point arithmetic on twisted Edwards curve.
 *
 * @details 
 * This file defines the struct of extended point and contains function 
 * prototypes of points arithmetic on twisted Edwards curve. 
 *******************************************************************************
 */

#ifndef _TEDCURVE_H
#define _TEDCURVE_H

#include "moncurve.h"

// Point in extended projective coordinates [x, y, z, e, h], e*h = t = x*y/z
typedef struct extended_point {
  __m256i x[NWORDS];  // extended x coordinate
  __m256i y[NWORDS];  // extended y coordinate
  __m256i z[NWORDS];  // extended z coordinate
  __m256i e[NWORDS];  // extended e coordinate (e*h = t)
  __m256i h[NWORDS];  // extended h coordinate (e*h = t)
} ExtPoint;

// function prototypes

void ted_point_add_avx2(ExtPoint *r, ExtPoint *p, ProPoint *q);
void ted_point_dbl_avx2(ExtPoint *r, ExtPoint *p);
void ted_point_query_table_avx2(ProPoint *r, const int pos, const __m256i b);
void ted_mul_fixbase_avx2(ProPoint *r, const __m256i *k);

#endif

