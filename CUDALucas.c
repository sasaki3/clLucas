char program[] = "CUDALucas v1.66";
/* CUDALucas.c
   Shoichiro Yamada Oct. 2010 

   This is an adaptation of Richard Crandall lucdwt.c, Sweeney MacLucasUNIX.c 
   and Guillermo Ballester Valor MacLucasFFTW.c code.
*/

/* Include Files */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <signal.h>
#ifdef linux
#include <sys/types.h>
#include <sys/stat.h>
#else
#include <direct.h>
#endif

// For clAmdFft Error check runtime_error()
#include <stdexcept>

#ifdef _MSC_VER
typedef struct timeval
{
  long tv_sec;
  long tv_usec;
} timeval;
int gettimeofday (struct timeval *tv, struct timezone *);
#else
#include <sys/time.h>
#include <unistd.h>
#endif

/* some definitions needed by mers package */
//#define MASK (0)
//#define SHIFT (0)
#define kErrLimit (0.35)
#define kErrChkFreq (100)
#define kErrChk (1)

/************************ definitions ************************************/
/* global variables needed */
double *two_to_phi, *two_to_minusphi;
double high, low, highinv, lowinv;
double Gsmall, Gbig, Hsmall, Hbig;
double bigAB, maxerr;
int *ip, quitting, checkpoint_iter, b, c, fftlen, s_f, t_f, v_f, r_f;
int threads, aggressive_f;
int g_err_flag;
int Nn;
char folder[132], input_filename[132];

#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>

#include <cmath>
#include <time.h>

#include "Kernels.cpp"

int
rdft (int n, int isgn, double *a, int *ip)
{
  void makect (int nc, int *ip, double *c);
  const int nc = n >> 2;
  int nw = ip[0];
  if (nw == 0)
    {
      makect (nc, ip, &a[n]);
      CHECK_OPENCL_ERROR (commandQueue.enqueueWriteBuffer (g_x, CL_TRUE, 0,
							   sizeof (double) *
							   (n / 4 * 5), a),
			  "Failed to write buffer.");
    }
  OPENCL_V_THROW (clAmdFftEnqueueTransform (plan,	// forward plan 
					    CLFFT_BACKWARD,	// forward 
					    1,	// 1 queue
					    &commandQueue (),
					    0, NULL, NULL, &g_x (), &g_x (),
					    NULL),
		  "Failed to clAmdFftEnqueueTransform.");
  clLucas.mul_runCLKernels ();
  OPENCL_V_THROW (clAmdFftEnqueueTransform (plan,	// forward plan
					    CLFFT_BACKWARD,	// forward
					    1,	// 1 queue
					    &commandQueue (),
					    0, NULL, NULL, &g_x (), &g_x (),
					    NULL),
		  "Failed to clAmdFftEnqueueTransform.");
  return 0;
}

/* -------- initializing routines -------- */
void
makect (int nc, int *ip, double *c)
{
  int j;
  const int nch = nc >> 1;
  double delta;
  ip[0] = 1;
  ip[1] = nc;
  if (nc > 1)
    {
      delta = atan (1.0) / nch;
      c[0] = cos (delta * nch);
      c[nch] = 0.5 * c[0];
      for (j = 1; j < nch; j++)
	{
	  c[j] = 0.5 * cos (delta * j);
	  c[nc - j] = 0.5 * sin (delta * j);
	}
    }
}

/**************************************************************
 *
 *      FFT and other related Functions
 *
 **************************************************************/
/* rint is not ANSI compatible, so we need a definition for 
 * WIN32 and other platforms with rint.
 * Also we use that to write the trick to rint()
 */
#define RINT_x86(x) (floor(x+0.5))
/****************************************************************************
 *           Lucas Test - specific routines                                 *
 ***************************************************************************/
int
init_lucas (double *x, int q, int n)
{
  cl_int status = 0;
  int j, qn, a, i, done;
  int size0, bj;
  double log2 = log (2.0);
  double ttp, ttmp;
  double *s_ttp, *s_ttmp;
  float *s_inv;
  float *s_inv2;
  float *s_inv3;
  double *s_ttp2, *s_ttmp2;
  double *s_ttp3, *s_ttmp3;
  double *s_ttmpp;
  double s_err;

  clLucas.fft_setup ((int) n / 2);

  two_to_phi = (double *) malloc (sizeof (double) * (n / 2));
  two_to_minusphi = (double *) malloc (sizeof (double) * (n / 2));
  s_inv = (float *) malloc (sizeof (float) * (n));
  s_ttp = (double *) malloc (sizeof (double) * (n));
  s_ttmp = (double *) malloc (sizeof (double) * (n));
  s_ttmpp = (double *) malloc (sizeof (double) * (n));
  s_inv2 = (float *) malloc (sizeof (float) * (n / threads));
  s_ttp2 = (double *) malloc (sizeof (double) * (n / threads));
  s_ttmp2 = (double *) malloc (sizeof (double) * (n / threads));
  s_inv3 = (float *) malloc (sizeof (float) * (n / threads));
  s_ttp3 = (double *) malloc (sizeof (double) * (n / threads));
  s_ttmp3 = (double *) malloc (sizeof (double) * (n / threads));

  g_x =
    cl::Buffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof (double) * (n / 4 * 5), x, &status);
  CHECK_OPENCL_ERROR (status, "Failed to created write buffer(g_x).");
  g_err =
    cl::Buffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof (double), &s_err, &status);
  CHECK_OPENCL_ERROR (status, "Failed to created write buffer(g_err).");
  g_carry =
    cl::Buffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof (int) * n / threads, x, &status);
  CHECK_OPENCL_ERROR (status, "Failed to created write buffer(g_carry).");
  g_inv =
    cl::Buffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof (float) * n, x, &status);
  CHECK_OPENCL_ERROR (status, "Failed to created write buffer(g_inv).");
  g_ttp =
    cl::Buffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof (double) * n, x, &status);
  CHECK_OPENCL_ERROR (status, "Failed to created write buffer(g_ttp).");
  g_ttmp =
    cl::Buffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof (double) * n, x, &status);
  CHECK_OPENCL_ERROR (status, "Failed to created write buffer(g_ttmp).");
  g_ttmpp =
    cl::Buffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof (double) * n, x, &status);
  CHECK_OPENCL_ERROR (status, "Failed to created write buffer(g_ttmpp).");
  g_inv2 =
    cl::Buffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof (float) * n / threads, x, &status);
  CHECK_OPENCL_ERROR (status, "Failed to created write buffer(g_inv2).");
  g_ttp2 =
    cl::Buffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof (double) * n / threads, x, &status);
  CHECK_OPENCL_ERROR (status, "Failed to created write buffer(g_ttp2).");
  g_ttmp2 =
    cl::Buffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof (double) * n / threads, x, &status);
  CHECK_OPENCL_ERROR (status, "Failed to created write buffer(g_ttmp2).");
  g_inv3 =
    cl::Buffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof (float) * n / threads, x, &status);
  CHECK_OPENCL_ERROR (status, "Failed to created write buffer(g_inv3).");
  g_ttp3 =
    cl::Buffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof (double) * n / threads, x, &status);
  CHECK_OPENCL_ERROR (status, "Failed to created write buffer(g_ttp3).");
  g_ttmp3 =
    cl::Buffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof (double) * n / threads, x, &status);
  CHECK_OPENCL_ERROR (status, "Failed to created write buffer(g_ttmp3).");
  s_err = 0;
  CHECK_OPENCL_ERROR (commandQueue.
		      enqueueWriteBuffer (g_err, CL_TRUE, 0, sizeof (double),
					  &s_err),
		      "Failed to write buffer(g_err).");
  low = floor ((exp (floor ((double) q / n) * log2)) + 0.5);
  high = low + low;
  lowinv = 1.0 / low;
  highinv = 1.0 / high;
  b = q % n;
  c = n - b;
  two_to_phi[0] = 1.0;
  two_to_minusphi[0] = 1.0 / (double) (n);
  qn = (b * 2) % n;
  for (i = 1, j = 2; j < n; j += 2, i++)
    {
      a = n - qn;
      two_to_phi[i] = exp (a * log2 / n);
      two_to_minusphi[i] = 1.0 / (two_to_phi[i] * n);
      qn += b * 2;
      qn %= n;
    }
  Hbig = exp (c * log2 / n);
  Gbig = 1 / Hbig;
  done = 0;
  j = 0;
  while (!done)
    {
      if (!((j * b) % n >= c || j == 0))
	{
	  a = n - ((j + 1) * b) % n;
	  i = n - (j * b) % n;
	  Hsmall = exp (a * log2 / n) / exp (i * log2 / n);
	  Gsmall = 1 / Hsmall;
	  done = 1;
	}
      j++;
    }
  bj = n;
  size0 = 1;
  bj = n - 1 * b;
  for (j = 0, i = 0; j < n; j = j + 2, i++)
    {
      ttmp = two_to_minusphi[i];
      ttp = two_to_phi[i];
      bj += b;
      bj = bj % n;
      size0 = (bj >= c);
      if (j == 0)
	size0 = 1;
      s_ttmp[j] = ttmp * 2.0;
      s_ttmpp[j] = ttmp * n;
      if (size0)
	{
	  s_inv[j] = (float) highinv;
	  ttmp *= Gbig;
	  s_ttp[j] = ttp * high;
	  ttp *= Hbig;
	}
      else
	{
	  s_inv[j] = (float) lowinv;
	  ttmp *= Gsmall;
	  s_ttp[j] = ttp * low;
	  ttp *= Hsmall;
	}
      s_ttmpp[j] *= s_ttp[j];
      bj += b;
      bj = bj % n;
      size0 = (bj >= c);
      if (j == (n - 2))
	size0 = 0;
      s_ttmp[j + 1] = ttmp * -2.0;
      s_ttmpp[j + 1] = ttmp * n;
      if (size0)
	{
	  s_inv[j + 1] = (float) highinv;
	  s_ttp[j + 1] = ttp * high;
	}
      else
	{
	  s_inv[j + 1] = (float) lowinv;
	  s_ttp[j + 1] = ttp * low;
	}
      s_ttmpp[j + 1] *= s_ttp[j + 1];
    }
  CHECK_OPENCL_ERROR (commandQueue.
		      enqueueWriteBuffer (g_inv, CL_TRUE, 0,
					  sizeof (float) * n, s_inv),
		      "Failed to write buffer(g_inv).");
  CHECK_OPENCL_ERROR (commandQueue.
		      enqueueWriteBuffer (g_ttmp, CL_TRUE, 0,
					  sizeof (double) * n, s_ttmp),
		      "Failed to write buffer(g_ttmp).");
  CHECK_OPENCL_ERROR (commandQueue.
		      enqueueWriteBuffer (g_ttmpp, CL_TRUE, 0,
					  sizeof (double) * n, s_ttmpp),
		      "Failed to write buffer(g_ttmpp).");

  for (i = 0, j = 0; i < n; i++)
    {
      if ((i % threads) == 0)
	{
	  s_inv2[j] = s_inv[i];
	  s_ttp2[j] = s_ttp[i];
	  s_ttmp2[j] = s_ttmp[i] * 0.5 * n;
	  s_inv3[j] = s_inv[i + 1];
	  s_ttp3[j] = s_ttp[i + 1];
	  s_ttmp3[j] = s_ttmp[i + 1] * (-0.5) * n;
	  j++;
	}
    }
  for (i = 0, j = 0; i < n; i++)
    s_ttp[i] *= s_inv[i];
  CHECK_OPENCL_ERROR (commandQueue.
		      enqueueWriteBuffer (g_ttp, CL_TRUE, 0,
					  sizeof (double) * n, s_ttp),
		      "Failed to write buffer(g_ttp).");
  CHECK_OPENCL_ERROR (commandQueue.
		      enqueueWriteBuffer (g_inv2, CL_TRUE, 0,
					  sizeof (float) * n / threads,
					  s_inv2),
		      "Failed to write buffer(g_inv2).");
  CHECK_OPENCL_ERROR (commandQueue.
		      enqueueWriteBuffer (g_ttp2, CL_TRUE, 0,
					  sizeof (double) * n / threads,
					  s_ttp2),
		      "Failed to write buffer(g_ttp2).");
  CHECK_OPENCL_ERROR (commandQueue.
		      enqueueWriteBuffer (g_ttmp2, CL_TRUE, 0,
					  sizeof (double) * n / threads,
					  s_ttmp2),
		      "Failed to write buffer(g_ttmp2).");
  CHECK_OPENCL_ERROR (commandQueue.
		      enqueueWriteBuffer (g_inv3, CL_TRUE, 0,
					  sizeof (float) * n / threads,
					  s_inv3),
		      "Failed to write buffer(g_inv3).");
  CHECK_OPENCL_ERROR (commandQueue.
		      enqueueWriteBuffer (g_ttp3, CL_TRUE, 0,
					  sizeof (double) * n / threads,
					  s_ttp3),
		      "Failed to write buffer(g_ttp3).");
  CHECK_OPENCL_ERROR (commandQueue.
		      enqueueWriteBuffer (g_ttmp3, CL_TRUE, 0,
					  sizeof (double) * n / threads,
					  s_ttmp3),
		      "Failed to write buffer(g_ttmp3).");

  free ((char *) s_inv);
  free ((char *) s_ttp);
  free ((char *) s_ttmp);
  free ((char *) s_ttmpp);
  free ((char *) s_inv2);
  free ((char *) s_ttp2);
  free ((char *) s_ttmp2);
  free ((char *) s_inv3);
  free ((char *) s_ttp3);
  free ((char *) s_ttmp3);

  ip = (int *) malloc (((size_t) (2 + sqrt ((float) n / 2)) * sizeof (int)));
  ip[0] = 0;
}

void
close_lucas (double *x)
{
  free ((char *) x);
  free ((char *) two_to_phi);
  free ((char *) two_to_minusphi);
  free ((char *) ip);
  clLucas.fft_close ();
}

double
normalize (double *x, int N, int err_flag)
{
  int i, j, k, lastloop, bj, offset, newOffset;
  int size0;
  double hi = high, hiinv = highinv, lo = low, loinv = lowinv;
  double temp0, tempErr;
  double maxerr = 0.0, err = 0.0, ttmpSmall = Gsmall, ttmpBig =
    Gbig, ttmp, ttp, ttmpTemp;
  double carry, ttpSmall = Hsmall, ttpBig = Hbig;
  double A = bigAB, B = bigAB;
  carry = -2.0;			/* this is the -2 of the LL x*x - 2 */

  lastloop = N - 2;

  bj = N;
  size0 = 1;
  offset = 0;

  for (i = 0, j = 0; j < N; j += 2, i++)
    {
      ttmp = two_to_minusphi[i];
      ttp = two_to_phi[i];

      for (k = 1; k < 2; k++)
	{
	  temp0 = x[offset];
	  temp0 *= 2.0;
	  newOffset = j + k;
	  tempErr = RINT_x86 (temp0 * ttmp);
	  if (err_flag)
	    {
	      err = fabs (temp0 * ttmp - tempErr);
	      if (err > maxerr)
		maxerr = err;
	    }
	  temp0 = tempErr + carry;
	  if (size0)
	    {
	      temp0 *= hiinv;
	      carry = RINT_x86 (temp0);
	      bj += b;
	      ttmp *= ttmpBig;
	      if (bj >= N)
		bj -= N;
	      x[offset] = (temp0 - carry) * ttp * hi;
	      size0 = (bj >= c);
	      ttp *= ttpBig;
	    }
	  else
	    {
	      temp0 *= loinv;
	      carry = RINT_x86 (temp0);
	      bj += b;
	      ttmp *= ttmpSmall;
	      if (bj >= N)
		bj -= N;
	      x[offset] = (temp0 - carry) * ttp * lo;
	      size0 = (bj >= c);
	      ttp *= ttpSmall;
	    }
	  offset = newOffset;
	}
      temp0 = x[offset];
      temp0 *= -2.0;
      newOffset = j + 2;
      if (j == lastloop)
	size0 = 0;
      tempErr = RINT_x86 (temp0 * ttmp);
      if (err_flag)
	{
	  err = fabs (temp0 * ttmp - tempErr);
	  if (err > maxerr)
	    maxerr = err;
	}
      temp0 = tempErr + carry;
      if (size0)
	{
	  temp0 *= hiinv;
	  carry = RINT_x86 (temp0);
	  bj += b;
	  ttmp *= ttmpBig;
	  if (bj >= N)
	    bj -= N;
	  x[offset] = (temp0 - carry) * ttp * hi;
	  size0 = (bj >= c);
	  ttp *= ttpBig;
	}
      else
	{
	  temp0 *= loinv;
	  carry = RINT_x86 (temp0);
	  bj += b;
	  ttmp *= ttmpSmall;
	  if (bj >= N)
	    bj -= N;
	  x[offset] = (temp0 - carry) * ttp * lo;
	  size0 = (bj >= c);
	  ttp *= ttpSmall;
	}
      offset = newOffset;
    }
  bj = N;
  ttmp = two_to_minusphi[0];
  ttp = two_to_phi[0];

  k = 0;
  while (int (carry) != 0)
    {
      ttmpTemp = x[k] * ttmp * N;
      size0 = (bj >= c);
      bj += b;
      temp0 = (ttmpTemp + carry);
      if (bj >= N)
	bj -= N;
      if (size0)
	{
	  temp0 *= hiinv;
	  ttmp *= ttmpBig;
	  carry = RINT_x86 (temp0);
	  x[k] = (temp0 - carry) * ttp * hi;
	  ttp *= ttpBig;
	}
      else
	{
	  temp0 *= loinv;
	  ttmp *= ttmpSmall;
	  carry = RINT_x86 (temp0);
	  x[k] = (temp0 - carry) * ttp * lo;
	  ttp *= ttpSmall;
	}
      k++;
    }
  return (maxerr);
}

double
last_normalize (double *x, int N, int err_flag)
{
  int i, j, k, bj, size0;
  double hi = high, hiinv = highinv, lo = low, loinv = lowinv, temp0, tempErr;
  double err = 0.0, terr = 0.0, ttmpSmall = Gsmall, ttmpBig =
    Gbig, ttmp, carry;
  carry = -2.0;			/* this is the -2 of the LL x*x - 2 */
  bj = N;
  size0 = 1;
  for (j = 0, i = 0; j < N; j += 2, i++)
    {
      ttmp = two_to_minusphi[i];
      temp0 = x[j];
      temp0 *= 2.0;
      tempErr = RINT_x86 (temp0 * ttmp);
      if (err_flag)
	{
	  terr = fabs (temp0 * ttmp - tempErr);
	  if (terr > err)
	    err = terr;
	}
      temp0 = tempErr + carry;
      if (size0)
	{
	  temp0 *= hiinv;
	  carry = RINT_x86 (temp0);
	  bj += b;
	  ttmp *= ttmpBig;
	  if (bj >= N)
	    bj -= N;
	  x[j] = (temp0 - carry) * hi;
	  size0 = (bj >= c);
	}
      else
	{
	  temp0 *= loinv;
	  carry = RINT_x86 (temp0);
	  bj += b;
	  ttmp *= ttmpSmall;
	  if (bj >= N)
	    bj -= N;
	  x[j] = (temp0 - carry) * lo;
	  size0 = (bj >= c);
	}
      temp0 = x[j + 1];
      temp0 *= -2.0;

      if (j == N - 2)
	size0 = 0;
      tempErr = RINT_x86 (temp0 * ttmp);
      if (err_flag)
	{
	  terr = fabs (temp0 * ttmp - tempErr);
	  if (terr > err)
	    err = terr;
	}
      temp0 = tempErr + carry;
      if (size0)
	{
	  temp0 *= hiinv;
	  carry = RINT_x86 (temp0);
	  bj += b;
	  ttmp *= ttmpBig;
	  if (bj >= N)
	    bj -= N;
	  x[j + 1] = (temp0 - carry) * hi;
	  size0 = (bj >= c);
	}
      else
	{
	  temp0 *= loinv;
	  carry = RINT_x86 (temp0);
	  bj += b;
	  ttmp *= ttmpSmall;
	  if (bj >= N)
	    bj -= N;
	  x[j + 1] = (temp0 - carry) * lo;
	  size0 = (bj >= c);
	}
    }
  bj = N;
  k = 0;
  while (int (carry) != 0 && k < N)
    {
      size0 = (bj >= c);
      bj += b;
      temp0 = (x[k] + carry);
      if (bj >= N)
	bj -= N;
      if (size0)
	{
	  temp0 *= hiinv;
	  carry = RINT_x86 (temp0);
	  x[k] = (temp0 - carry) * hi;
	}
      else
	{
	  temp0 *= loinv;
	  carry = RINT_x86 (temp0);
	  x[k] = (temp0 - carry) * lo;
	}
      k++;
    }
  if (int (carry) != 0)
    err = 0.5;
  return (err);
}

double
lucas_square (double *x, int N, int iter, int last, int error_log)
{
  double terr;
  bigAB = 6755399441055744.0;
  terr = 0.0;
  rdft (N, 1, x, ip);
  if (iter == last)
    {
      CHECK_OPENCL_ERROR (commandQueue.
			  enqueueReadBuffer (g_x, CL_TRUE, 0,
					     sizeof (double) * N, x),
			  "Failed to read buffer(g_x).");
      terr = last_normalize (x, N, error_log);
    }
  else
    {
      if ((iter % checkpoint_iter) == 0)
	{
	  CHECK_OPENCL_ERROR (commandQueue.
			      enqueueReadBuffer (g_x, CL_TRUE, 0,
						 sizeof (double) * N, x),
			      "Failed to read buffer(g_x).");
	  terr = last_normalize (x, N, error_log);
	}

      g_err_flag = error_log;
      clLucas.normalize_runCLKernels ();
      clLucas.normalize2_runCLKernels ();

      {
	double l_err;
	if (!aggressive_f)
	  clFinish (commandQueue ());
      }
      if (error_log)
	{
	  double *c_err = (double *) malloc (sizeof (double));
	  CHECK_OPENCL_ERROR (commandQueue.
			      enqueueReadBuffer (g_err, CL_TRUE, 0,
						 sizeof (double), c_err),
			      "Failed to read buffer(g_err).");
	  terr = fmax (c_err[0], terr);
	  free (c_err);
	}
    }
  return (terr);
}

/* Choose the lenght of FFT , n is the power of two preliminary solution,
N is the minimum required length, the return is the estimation of optimal 
lenght.

The estimation is made very rougly. I suposse a prime k pass cost about
      k*lengthFFT cpu time (in some units)
*/
int
choose_length (int input_length)
{
#ifdef TEST
  printf ("FFT selector called on %d\n", input_length);
#endif
  int np[18] =
    { 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30 };
  int output_length = 1;
  int i, tmp;
  do
    {
#ifdef TEST
      printf ("Output_length is now %d\n", output_length);
#endif
      for (i = 0; i < 18; i++)
	{
	  tmp = output_length * np[i];
#ifdef TEST
	  printf ("Output_length * np[%d] is %d\n", i, tmp);
#endif
	  if (tmp >= input_length)
	    {
#ifdef TEST
	      printf
		("%d is greater than input %d, returning %d, which is %dK + %d\n",
		 tmp, input_length, tmp, tmp / 1024, tmp % 1024);
#endif
	      return (int) tmp;
	    }
	}
    }
  while (output_length *= 2);
  return 0;
}

//From apsen
void
print_time_from_seconds (int sec)
{
  if (sec > 3600)
    {
      printf ("%d", sec / 3600);
      sec %= 3600;
      printf (":%02d", sec / 60);
    }
  else
    printf ("%d", sec / 60);
  sec %= 60;
  printf (":%02d", sec);
}

int
is_big2 (int j, int big, int small, int n)
{
  return ((((big * j) % n) >= small) || j == 0);
}

void
balancedtostdrep (double *x, int n, int b, int c, double hi, double lo,
		  int mask, int shift)
{
  int sudden_death = 0, j = 0, NminusOne = n - 1, k, k1;
  while (1)
    {
      k = j + ((j & mask) >> shift);
      if (x[k] < 0.0)
	{
	  k1 = (j + 1) % n;
	  k1 += (k1 & mask) >> shift;
	  --x[k1];
	  if (j == 0 || (j != NminusOne && is_big2 (j, b, c, n)))
	    x[k] += hi;
	  else
	    x[k] += lo;
	}
      else if (sudden_death)
	break;
      if (++j == n)
	{
	  sudden_death = 1;
	  j = 0;
	}
    }
}

int
is_zero (double *x, int n, int mask, int shift)
{
  int j, offset;
  for (j = 0; j < n; ++j)
    {
      offset = j + ((j & mask) >> shift);
      if (int (x[offset]))
	return (0);
    }
  return (1);
}

void
printbits (double
	   *x,
	   int q,
	   int N,
	   int b, int c, double high, double low, int totalbits,
	   int flag, char *expectedResidue)
{
  char *bits = (char *) malloc ((int) totalbits);
  char residue[32];
  char temp[32];
  int j, k, i, word;
  FILE *fp;
  if (flag)
    {
      fp = fopen ("result.txt", "a");
      if (fp == NULL)
	{
	  fprintf (stderr, "Cannot write results to result.txt\n");
	  exit (1);
	}
    }
  if (is_zero (x, N, 0, 0))
    {
      printf ("M( %d )P, n = %d, %s", q, N, program);
      if (flag)
	{
	  fprintf (fp, "M( %d )P, n = %d, %s", q, N, program);
	  fprintf (fp, "\n");
	  fclose (fp);
	}
    }
  else
    {
      double *x_tmp;
      x_tmp = (double *) malloc (sizeof (double) * N);
      for (i = 0; i < N; i++)
	x_tmp[i] = x[i];
      balancedtostdrep (x_tmp, N, b, c, high, low, 0, 0);
      printf ("M( %d )C, 0x", q);
      if (flag)
	fprintf (fp, "M( %d )C, 0x", q);
      j = 0;
      i = 0;
      do
	{
	  k = (int) (ceil ((double) q * (j + 1) / N) -
		     ceil ((double) q * j / N));
	  if (k > totalbits)
	    k = totalbits;
	  totalbits -= k;
	  word = (int) x_tmp[j++];
	  while (k--)
	    {
	      bits[i++] = (char) ('0' + (word & 1));
	      word >>= 1;
	    }
	}
      while (totalbits);
      sprintf (residue, "");
      while (i)
	{
	  k = 0;
	  for (j = 0; j < 4; j++)
	    {
	      i--;
	      k <<= 1;
	      if (bits[i] == '1')
		k++;
	    }
	  if (k > 9)
	    {
	      sprintf (temp, "%s", residue);
	      sprintf (residue, "%s%c", temp, (char) ('a' + k - 10));
	    }
	  else
	    {
	      sprintf (temp, "%s", residue);
	      sprintf (residue, "%s%c", temp, (char) ('0' + k));
	    }
	}
      free (x_tmp);
      printf ("%s", residue);
      printf (", n = %d, %s", N, program);
      if (flag)
	{
	  fprintf (fp, "%s", residue);
	  fprintf (fp, ", n = %d, %s", N, program);
	  fprintf (fp, "\n");
	  fclose (fp);
	}
      if (expectedResidue && strcmp (residue, expectedResidue))
	printf
	  ("Expected residue [%s] does not match actual residue [%s]\n",
	   expectedResidue, residue);
    }
  free (bits);
}

void
rm_checkpoint (int q)
{
  char chkpnt_cfn[32];
  char chkpnt_tfn[32];
  sprintf (chkpnt_cfn, "c" "%d", q);
  sprintf (chkpnt_tfn, "t" "%d", q);
  (void) unlink (chkpnt_cfn);
  (void) unlink (chkpnt_tfn);
}

double *
read_checkpoint (int q, int *n, int *j)
{
  FILE *fPtr;
  int q_r, n_r, j_r;
  double *x;
  char chkpnt_cfn[32];
  char chkpnt_tfn[32];
  sprintf (chkpnt_cfn, "c" "%d", q);
  sprintf (chkpnt_tfn, "t" "%d", q);
  fPtr = fopen (chkpnt_cfn, "rb");
  if (!fPtr)
    {
      fPtr = fopen (chkpnt_tfn, "rb");
      if (!fPtr)
	return NULL;
    }
  // check parameters
  if (fread (&q_r, 1, sizeof (q_r), fPtr) != sizeof (q_r)
      || fread (&n_r, 1, sizeof (n_r), fPtr)
      != sizeof (n_r) || fread (&j_r, 1, sizeof (j_r), fPtr) != sizeof (j_r))
    {
      fprintf (stderr,
	       "\nThe checkpoint doesn't match current test.  Current test will be restarted\n");
      fclose (fPtr);
      return NULL;
    }
  if (q != q_r)
    {
      fprintf
	(stderr,
	 "\nThe checkpoint doesn't match current test.  Current test will be restarted\n");
      fclose (fPtr);
      return NULL;
    }
  // check for successful read of z, delayed until here since zSize can vary
  x = (double *) malloc (sizeof (double) * (n_r + n_r));
  if (fread (x, 1, sizeof (double) * (n_r), fPtr) !=
      (sizeof (double) * (n_r)))
    {
      fprintf (stderr,
	       "\nThe checkpoint doesn't match current test.  Current test will be restarted\n");
      fclose (fPtr);
      free (x);
      return NULL;
    }
  // have good stuff, do checkpoint
  *n = n_r;
  *j = j_r;
  fclose (fPtr);
  return x;
}

void
write_checkpoint (double *x, int q, int n, int j)
{
  FILE *fPtr;
  char chkpnt_cfn[32];
  char chkpnt_tfn[32];
  sprintf (chkpnt_cfn, "c" "%d", q);
  sprintf (chkpnt_tfn, "t" "%d", q);
  (void) unlink (chkpnt_tfn);
  (void) rename (chkpnt_cfn, chkpnt_tfn);
  fPtr = fopen (chkpnt_cfn, "wb");
  if (!fPtr)
    return;
  fwrite (&q, 1, sizeof (q), fPtr);
  fwrite (&n, 1, sizeof (n), fPtr);
  fwrite (&j, 1, sizeof (j), fPtr);
  fwrite (x, 1, sizeof (double) * n, fPtr);
  fclose (fPtr);
  if (s_f)			// save all chekpoint file
    {
      char chkpnt_sfn[32];
#ifdef linux
      sprintf (chkpnt_sfn, "%s/s" "%d.%d", folder, q, j);
#else
      sprintf (chkpnt_sfn, "%s\\s" "%d.%d", folder, q, j);
#endif
      fPtr = fopen (chkpnt_sfn, "wb");
      if (!fPtr)
	return;
      fwrite (&q, 1, sizeof (q), fPtr);
      fwrite (&n, 1, sizeof (n), fPtr);
      fwrite (&j, 1, sizeof (j), fPtr);
      fwrite (x, 1, sizeof (double) * n, fPtr);
      fclose (fPtr);
    }
}

void
SetQuitting (int sig)
{
  quitting = 1;
  printf ("^C caught.  Writing checkpoint.\n");
}

#ifdef WIN32
BOOL WINAPI
HandlerRoutine (DWORD
		/*dwCtrlType */ )
{
  SetQuitting (1);
  return true;
}
#endif

/**************************************************************
 *
 *      Main Function
 *
 **************************************************************/
int
check (int q, char *expectedResidue)
{
  int n, j = 1L, last = 2L, flag;
  size_t k;
  double terr, *x = NULL;
  int restarting = 0;
  timeval time0, time1;
  if (!expectedResidue)
    {
#ifdef WIN32
      SetConsoleCtrlHandler (HandlerRoutine, TRUE);
#else
      // We log to file in most cases anyway.
      signal (SIGTERM, SetQuitting);
      signal (SIGINT, SetQuitting);
#endif
    }
  n = q / 20;
  do
    {				/* while (restarting) */
      n = choose_length (n);
      maxerr = 0.0;
      if (fftlen)
	n = fftlen;
      if ((n / threads) > 65535)
	{
	  printf ("over specifications Grid = %d\n", (int) n / threads);
	  exit (2);
	}
      if (!expectedResidue && !restarting
	  && (x = read_checkpoint (q, &n, &j)) != NULL)
	printf
	  ("\ncontinuing work from a partial result M%d fft length = %d iteration = %d\n",
	   q, n, j);
      else
	{
	  if (!expectedResidue)
	    printf ("\nstart M%d fft length = %d\n", q, n);
	  x = (double *) malloc (sizeof (double) * (n + n));
	  for (k = 1; k < n; k++)
	    x[k] = 0.0;
	  x[0] = 4.0;
	  j = 1;
	}
      fflush (stdout);
      restarting = 0;

      Nn = n;

      init_lucas (x, q, n);

      gettimeofday (&time0, NULL);
      last = q - 2;		/* the last iteration done in the primary loop */
      for (; !restarting && j <= last; j++)
	{
	  if ((j % kErrChkFreq) == 1 || j < 1000 || t_f)
	    flag = kErrChk;
	  else
	    flag = 0;
	  terr = lucas_square (x, n, j, last, flag);

	  if (flag == kErrChk && terr > maxerr)
	    maxerr = terr;
	  if (terr > kErrLimit)
	    {			/* n is not big enough; increase it and start over */
	      if (fftlen)
		{
		  printf ("err = %g,fft length = %d exiting.\n",
			  (double) terr, (int) n);
		  exit (2);
		}
	      printf ("err = %g, increasing n from %d\n", (double) terr,
		      (int) n);
	      n++;
	      restarting = 1;
	    }
	  if ((j % checkpoint_iter) == 0)
	    {
	      gettimeofday (&time1, NULL);
	      printf ("Iteration %d ", j);
	      printbits (x, q, n, b, c, high, low, 64, 0, expectedResidue);
	      long diff = time1.tv_sec - time0.tv_sec;
	      long diff1 = 1000000 * diff + time1.tv_usec - time0.tv_usec;
	      printf (" err = %3.4g (", maxerr);
	      print_time_from_seconds (diff);
	      printf (" real, %3.4f ms/iter, ETA ",
		      diff1 / 1000.0 / checkpoint_iter);
	      diff = (long) ((last - j) / checkpoint_iter * (diff1 / 1e6));
	      print_time_from_seconds (diff);
	      printf (")\n");
	      fflush (stdout);
	      gettimeofday (&time0, NULL);
	      if (expectedResidue)
		j = last + 1;
	    }
	  if (((j % checkpoint_iter) == 0 || quitting == 1)
	      && !expectedResidue)
	    {
	      CHECK_OPENCL_ERROR (commandQueue.
				  enqueueReadBuffer (g_x, CL_TRUE, 0,
						     sizeof (double) * n, x),
				  "Failed to read buffer(g_x).");
	      write_checkpoint (x, q, n, j + 1);
	      if (quitting == 1)
		j = last + 1;
	    }
	}
      if (restarting == 0 && !expectedResidue && quitting == 0)
	{
	  printbits (x, q, n, b, c, high, low, 64, 1, 0);
	  printf ("\n");
	  fflush (stdout);
	  rm_checkpoint (q);
	}
      close_lucas (x);
    }
  while (restarting);
  return (0);
}

int
main (int argc, char *argv[])
{
  int q;
  int device_number = 0;

  checkpoint_iter = 10000;
  threads = 256;
  fftlen = 0;
  quitting = 0;
  aggressive_f = s_f = t_f = v_f = r_f = 0;

  if (argc == 1)
    {
      fprintf
	(stderr,
	 "Usage: %s [-d device_number] [-threads 32|64|128|256|512|1024] [-c checkpoint_iteration] [-f fft_length] [-s folder] [-t] [-aggressive] -r|exponent|input_filename\n",
	 argv[0]);
      fprintf (stderr,
	       "                       -threads set threads number(default=256)\n");
      fprintf (stderr, "                       -f set fft length\n");
      fprintf (stderr,
	       "                       -s save all checkpoint files\n");
      fprintf (stderr,
	       "                       -t check round off error all iterations\n");
      fprintf (stderr,
	       "                       -aggressive GPU aggressive(default polite)\n");
      exit (2);
    }

  while (argc > 1)
    {
      if (strcmp (argv[1], "-v") == 0)
	{
	  v_f = 1;
	  argv++;
	  argc--;
	}
      else if (strcmp (argv[1], "-t") == 0)
	{
	  t_f = 1;
	  argv++;
	  argc--;
	}
      else if (strcmp (argv[1], "-aggressive") == 0)
	{
	  aggressive_f = 1;
	  argv++;
	  argc--;
	}
      else if (strcmp (argv[1], "-r") == 0)
	{
	  r_f = 1;
	  argv++;
	  argc--;
	}
      else if (strcmp (argv[1], "-d") == 0)
	{
	  if (argc < 3)
	    {
	      fprintf (stderr, "can't parse -d option\n");
	      exit (2);
	    }
	  device_number = atoi (argv[2]);
	  clLucas.set_deviceId (device_number);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-threads") == 0)
	{
	  if (argc < 3)
	    {
	      fprintf (stderr, "can't parse -threads option\n");
	      exit (2);
	    }
	  threads = atoi (argv[2]);
	  if (threads != 32 && threads != 64 && threads != 128
	      && threads != 256 && threads != 512 && threads != 1024)
	    {
	      fprintf (stderr, "can't parse -threads option\n");
	      exit (2);
	    }
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-c") == 0)
	{
	  if (argc < 3)
	    {
	      fprintf (stderr, "can't parse -c option\n");
	      exit (2);
	    }
	  checkpoint_iter = atoi (argv[2]);
	  if (checkpoint_iter == 0)
	    {
	      fprintf (stderr, "can't parse -c option\n");
	      exit (2);
	    }
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-f") == 0)
	{
	  if (argc < 3)
	    {
	      fprintf (stderr, "can't parse -f option\n");
	      exit (2);
	    }
	  fftlen = atoi (argv[2]);
	  if (fftlen == 0)
	    {
	      fprintf (stderr, "can't parse -f option\n");
	      exit (2);
	    }
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-s") == 0)
	{
	  s_f = 1;
	  if (argc < 3)
	    {
	      fprintf (stderr, "can't parse -s option\n");
	      exit (2);
	    }
	  sprintf (folder, "%s", argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else
	{
	  q = atoi (argv[1]);
	  if (q == 0)
	    sprintf (input_filename, "%s", argv[1]);
	  argv++;
	  argc--;
	}
    }

//  init_device (device_number);
  clLucas.setup ();

  if (r_f)
    {
      checkpoint_iter = 10000;
      check (86243, "23992ccd735a03d9");
      check (132049, "4c52a92b54635f9e");
      check (216091, "30247786758b8792");
      check (756839, "5d2cbe7cb24a109a");
      check (859433, "3c4ad525c2d0aed0");
      check (1257787, "3f45bf9bea7213ea");
      check (1398269, "a4a6d2f0e34629db");
      check (2976221, "2a7111b7f70fea2f");
      check (3021377, "6387a70a85d46baf");
      check (6972593, "88f1d2640adb89e1");
      check (13466917, "9fdc1f4092b15d69");
      check (20996011, "5fc58920a821da11");
      check (24036583, "cbdef38a0bdc4f00");
      check (25964951, "62eb3ff0a5f6237c");
      check (30402457, "0b8600ef47e69d27");
      check (32582657, "02751b7fcec76bb1");
      check (37156667, "67ad7646a1fad514");
      check (42643801, "8f90d78d5007bba7");
      check (43112609, "e86891ebf6cd70c4");
    }
  else
    {
      if (s_f)
	{
#ifdef linux
	  mode_t mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
	  if (mkdir (folder, mode) != 0)
	    printf
	      ("mkdir: cannot create directory `%s': File exists\n", folder);
#else
	  if (_mkdir (folder) != 0)
	    printf
	      ("mkdir: cannot create directory `%s': File exists\n", folder);
#endif
	}
      if (q == 0)
	{
	  int i, currentLine;
	  FILE *fp, *fpi;
	  char str[132];
	  fp = fopen (input_filename, "r");
	  if (fp == NULL)
	    {
	      fprintf (stderr, "Cannot open '%s'\n", input_filename);
	      exit (2);
	    }
	  fpi = fopen ("cudalucas.ini", "r");
	  if (fpi != NULL)
	    {
	      if (fgets (str, 132, fpi) == NULL);
	      currentLine = atoi (str);
	      fclose (fpi);
	      printf ("Continue test of file '%s' at line %d\n",
		      input_filename, currentLine);
	    }
	  else
	    {
	      currentLine = 0;
	      printf ("Start test of file '%s'\n", input_filename);
	    }
	  for (i = 0; i < currentLine; ++i)
	    if (fgets (str, 132, fp) == NULL);
	  while (fgets (str, 132, fp) != NULL && quitting == 0)
	    {
	      if (sscanf (str, "%u", &q) == 1)
		{
		  if (q < 86243)
		    printf (" too small Exponent %d\n", q);
		  else
		    check (q, 0);
		}
	      if (quitting == 0)
		++currentLine;
	      fpi = fopen ("cudalucas.ini", "w");
	      if (fpi != NULL)
		{
		  fprintf (fpi, "%d\n", currentLine);
		  fclose (fpi);
		}
	    }
	  fclose (fp);
	}
      else
	{
	  if (q < 86243)
	    printf (" too small Exponent %d\n", q);
	  else
	    check (q, 0);
	}
    }
}
