/**
 *******************************************************************************
 * @file tedcurve.c
 * @version 1.0.1
 * @date 2020-09-01
 * @copyright Copyright © 2020 by University of Luxembourg.
 * @author Developed at SnT APSIA by: Hao Cheng, Johann Groszschaedl and Jiaqi Tian.
 *
 * @brief C source file of point arithmetic on twisted Edward curve.
 *
 * @details 
 * This file contains (4*1)-way parallel point operations on twisted Edwards curve. 
 *******************************************************************************
 */

#include "base.h"
#include "tedcurve.h"

// "1/2" in the field
static const uint64_t one_half[4] = { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 
  0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF };

/**
 * @brief Point addition.
 *
 * @details
 * Unified mixed addition R = P + Q on a twisted Edwards curve with a = -1.
 *
 * @param r Point in extended projective coordinates [x, y, z, e, h], e*h = t = x*y/z
 * @param p Point in extended projective coordinates [x, y, z, e, h], e*h = t = x*y/z
 * @param q Point in Duif representation [(y+x)/2, (y-x)/2, d*x*y]
 */
void ted_point_add_avx2(ExtPoint *r, ExtPoint *p, ProPoint *q)
{
  __m256i t[NWORDS];

  mpi29_gfp_mul_avx2(t, p->e, p->h);
  mpi29_gfp_sub_avx2(r->e, p->y, p->x);
  mpi29_gfp_add_avx2(r->h, p->y, p->x);
  mpi29_gfp_mul_avx2(r->x, r->e, q->y);
  mpi29_gfp_mul_avx2(r->y, r->h, q->x);
  mpi29_gfp_sub_avx2(r->e, r->y, r->x);
  mpi29_gfp_add_avx2(r->h, r->y, r->x);
  mpi29_gfp_mul_avx2(r->x, t, q->z);
  mpi29_gfp_sbc_avx2(t, p->z, r->x);
  mpi29_gfp_add_avx2(r->x, p->z, r->x);
  mpi29_gfp_mul_avx2(r->z, t, r->x);
  mpi29_gfp_mul_avx2(r->y, r->x, r->h);
  mpi29_gfp_mul_avx2(r->x, r->e, t);
}


/**
 * @brief Point doubling.
 *
 * @details
 * Doubling R = 2*P on a twisted Edwards curve with a = -1.
 *
 * @param r Point in extended projective coordinates [x, y, z, e, h], e*h = t = x*y/z
 * @param p Point in extended projective coordinates [x, y, z, e, h], e*h = t = x*y/z
 */
void ted_point_dbl_avx2(ExtPoint *r, ExtPoint *p)
{
  __m256i t[NWORDS];

  mpi29_gfp_sqr_avx2(r->e, p->x);
  mpi29_gfp_sqr_avx2(r->h, p->y);
  mpi29_gfp_sbc_avx2(t, r->e, r->h);
  mpi29_gfp_add_avx2(r->h, r->e, r->h);
  mpi29_gfp_add_avx2(r->x, p->x, p->y);
  mpi29_gfp_sqr_avx2(r->e, r->x);
  mpi29_gfp_sub_avx2(r->e, r->h, r->e);
  mpi29_gfp_sqr_avx2(r->y, p->z);
  mpi29_gfp_mul29_avx2(r->y, r->y, 2);
  mpi29_gfp_add_avx2(r->y, t, r->y);
  mpi29_gfp_mul_avx2(r->x, r->e, r->y);
  mpi29_gfp_mul_avx2(r->z, r->y, t);
  mpi29_gfp_mul_avx2(r->y, t, r->h);
}


/**
 * @brief Initialize a point with extened projective coordinates.
 *
 * @details
 * Initialize a point P to be [0, 1, 1, 0, 1].
 * 
 * @param p Extented projective point
 */
static void ted_point_init_ext_avx2(ExtPoint *p)
{
  const __m256i zero = VZERO;
  const __m256i one = VSET164(1);

  p->x[0] = p->e[0] = zero;
  p->y[0] = p->z[0] = p->h[0] = one;
  p->x[1] = p->x[2] = p->x[3] = p->x[4] = zero;
  p->x[5] = p->x[6] = p->x[7] = p->x[8] = zero;
  p->y[1] = p->y[2] = p->y[3] = p->y[4] = zero;
  p->y[5] = p->y[6] = p->y[7] = p->y[8] = zero;
  p->z[1] = p->z[2] = p->z[3] = p->z[4] = zero;
  p->z[5] = p->z[6] = p->z[7] = p->z[8] = zero;
  p->e[1] = p->e[2] = p->e[3] = p->e[4] = zero;
  p->e[5] = p->e[6] = p->e[7] = p->e[8] = zero;
  p->h[1] = p->h[2] = p->h[3] = p->h[4] = zero;
  p->h[5] = p->h[6] = p->h[7] = p->h[8] = zero;
}

/**
 * @brief Point multiplication based on the look-up table.
 *
 * @details
 * Look up the table with specifying the position to obtain the multiple of base
 * point (in Duif representation).
 *
 * @param r Point of the table in Duif representation [(y+x)/2, (y-x)/2, d*x*y]
 * @param pos Position of the table
 * @param b Scalar (a nibble)
 */
void ted_point_query_table_avx2(ProPoint *r, const int pos, const __m256i b)
{
  const __m256i babs  = VABS8(b);   // the abs of scalar nibble
  const __m256i one   = VSET164(1);
  const __m256i zero  = VZERO; 
  __m256i mask[9], xP[4], yP[4], zP[4], t[NWORDS];
  __m256i xcoor, ycoor, zcoor, index, tmp, bsign, bmask;
  int i, j;

  // create the masks
  index = zero;
  for (i = 0; i < 9; i++) {
    mask[i] = VXOR(babs, index);
    mask[i] = VSUB(mask[i], one);
    mask[i] = VSHR(mask[i], 32);  
    mask[i] = VSHUF32(mask[i], 0xA0);
    index = VADD(index, one);    
  }

  // query the table 
  for (i = 0; i < 4; i++) {
    // P is [0, 0, 0] now
    xP[i] = yP[i] = zP[i] = zero; 

    xcoor = VSET164(one_half[i]);
    ycoor = VSET164(one_half[i]);

    xP[i] = VXOR(xP[i], VAND(mask[0], xcoor));
    yP[i] = VXOR(yP[i], VAND(mask[0], ycoor));

    for (j = 0; j < 8; j++) {
      // Using SET164 is pretty slow here 
      xcoor = VBROAD64(VLOAD128(&(base[pos][j].x[i]))); 
      ycoor = VBROAD64(VLOAD128(&(base[pos][j].y[i])));
      zcoor = VBROAD64(VLOAD128(&(base[pos][j].z[i])));

      xP[i] = VXOR(xP[i], VAND(mask[j+1], xcoor));
      yP[i] = VXOR(yP[i], VAND(mask[j+1], ycoor));
      zP[i] = VXOR(zP[i], VAND(mask[j+1], zcoor));
    } 
  }

  // if b < 0, bsign is 1; if b >= 0, bsign is 0.
  bsign = VSHR(b, 7);
  bmask = VSUB(zero, bsign);
  for (i = 0; i < 4; i++) {
    tmp   = VAND(VXOR(xP[i], yP[i]), bmask);
    xP[i] = VXOR(xP[i], tmp);
    yP[i] = VXOR(yP[i], tmp);
  }

  lut_conv_coor2mpi29_avx2(r->x, xP);
  lut_conv_coor2mpi29_avx2(r->y, yP);
  lut_conv_coor2mpi29_avx2(r->z, zP);

  mpi29_copy_avx2(t, r->z);
  for (i = 0; i < NWORDS; i++) mask[i] = zero;
  mpi29_gfp_sub_avx2(t, mask, t);
  mpi29_cswap_avx2(r->z, t, bsign);
}


/**
 * @brief Convert a scalar to signed nibbles.
 *
 * @details
 * Convert the 256-bit scalar to 64 signed nibbles and store them in an array.
 *
 * @param e Nibbles
 * @param k Scalar
 */
void ted_conv_scalar2nibble_avx2(__m256i *e, const __m256i *k)
{
  int i;
  const __m256i eight = VSET164(8);
  const __m256i mask4 = VSET164(0x0F);
  const __m256i mask8 = VSET164(0xFF);
  __m256i carry = VZERO;

  // convert scalar to nibbles
  for (i = 0; i < 8; i++) {
    e[8*i] = VAND(k[i], mask4);
    e[8*i+1] = VAND(VSHR(k[i], 4), mask4);
    e[8*i+2] = VAND(VSHR(k[i], 8), mask4);
    e[8*i+3] = VAND(VSHR(k[i], 12), mask4);
    e[8*i+4] = VAND(VSHR(k[i], 16), mask4);
    e[8*i+5] = VAND(VSHR(k[i], 20), mask4);
    e[8*i+6] = VAND(VSHR(k[i], 24), mask4);
    e[8*i+7] = VAND(VSHR(k[i], 28), mask4);
  }

  // convert unsigned nibbles to signed
  for (i = 0; i < 63; i++) {
    e[i] = VADD(e[i], carry);
    carry = VADD(e[i], eight);
    carry = VSHR(carry, 4);
    e[i] = VSUB(e[i], VSHL(carry, 4));
    e[i] = VAND(e[i], mask8);
  }
  e[63] = VADD(e[63], carry);
  e[63] = VAND(e[63], mask8);
}


/**
 * @brief Fixed-base scalar multiplication on twisted Edwards curve.
 *
 * @details
 * R = k * B.
 * Compute a scalar multiplication R = k * B with a fixed base B (x, 4/5) on 
 * twisted Edwards curve.
 * 
 * @param r Projective point 
 * @param k Scalar 
 */
void ted_mul_fixbase_avx2(ProPoint *r, const __m256i *k)
{
  ExtPoint h;
  __m256i e[64], carry, kp[8];
  const __m256i t0 = VSET164(0xFFFFFFF8U);
  const __m256i t1 = VSET164(0x7FFFFFFFU);
  const __m256i t2 = VSET164(0x40000000U);
  int i;

  // prune scalar k
  for (i = 0; i < 8; i++) kp[i] = k[i];
  kp[0] = VAND(kp[0], t0);
  kp[7] = VAND(kp[7], t1);
  kp[7] = VOR(kp[7], t2);

  ted_conv_scalar2nibble_avx2(e, kp);

  ted_point_init_ext_avx2(&h);

  for (i = 1; i < 64; i += 2) {
    ted_point_query_table_avx2(r, i>>1, e[i]);
    ted_point_add_avx2(&h, &h, r);
  }

  ted_point_dbl_avx2(&h, &h);
  ted_point_dbl_avx2(&h, &h);
  ted_point_dbl_avx2(&h, &h);
  ted_point_dbl_avx2(&h, &h);

  for (i = 0; i < 64; i += 2) {
    ted_point_query_table_avx2(r, i>>1, e[i]);
    ted_point_add_avx2(&h, &h, r);
  }

  // mpi29_copy_avx2(r->x, h.x);
  mpi29_copy_avx2(r->y, h.y);
  mpi29_copy_avx2(r->z, h.z);
}
