#ifndef INSTANCE643_H
#define INSTANCE643_H

/****************************************************************
 * Copyright (c) 2020~2020, 18-643 Course Staff, CMU
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:

 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.

 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.

 * The views and conclusions contained in the software and
 * documentation are those of the authors and should not be
 * interpreted as representing official policies, either expressed or
 * implied, of the FreeBSD Project.
 ****************************************************************/

/*
 * CMU 18643 Fall 2020 Lab Exercise
 *
 * The parameters in this file sets the problem sizes
 *
 */

typedef float cnndata_t;

#define BATCH_SIZE 10

#if 1
/* 
 * weights parameters
 */
#define K_WTS (3) // weight width and height (square)
                   // same depth as output
#define S_WTS (1) // sliding stride

/* 
 * output feature map paramters
 */
#define R_OFM (13) // height
#define C_OFM (13) // width
#define M_OFM (128) // depth

/*
 * input feature map paramters
 */
#define N_IFM (192) // depth

#else

#define K_WTS (4) // weight width and height (square)
#define S_WTS (1) // sliding stride

#define R_OFM (16) // height
#define C_OFM (16) // width
#define M_OFM (128) // depth

#define N_IFM (128) // depth

#endif

#define R_IFM (R_OFM*S_WTS+K_WTS-S_WTS) // derived height
#define C_IFM (C_OFM*S_WTS+K_WTS-S_WTS) // derived width

#endif
