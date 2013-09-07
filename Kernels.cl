/* http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html
   base code from Takuya OOURA.  */
__kernel void mul_Kernel (const int n, __global double *a)
{
  double wkr, wki, xr, xi, yr, yi, cc, d, aj, aj1, ak, ak1;
  int m = n >> 1;
  int nc = n >> 2;
  int threadID = get_global_id (0);
  int j = threadID << 1;
  int j2 = threadID;
  //c = &a[n];
  if (threadID)
    {
      wkr = 0.5 - a[nc - j2+n];
      wki = a[j2+n];
      aj = a[j];
      aj1 = a[1 + j];
      ak = a[n - j];
      ak1 = a[1 + n - j];
      xr = aj - ak;
      xi = aj1 + ak1;
      yr = wkr * xr - wki * xi;
      yi = wkr * xi + wki * xr;
      aj -= yr;
      aj1 -= yi;
      ak += yr;
      ak1 -= yi;
      cc = aj;
      d = -aj1;
      aj1 = -2.0 * cc * d;
      aj = (cc + d) * (cc - d);
      cc = ak;
      d = -ak1;
      ak1 = -2.0 * cc * d;
      ak = (cc + d) * (cc - d);
      xr = aj - ak;
      xi = aj1 + ak1;
      yr = wkr * xr + wki * xi;
      yi = wkr * xi - wki * xr;
      aj -= yr;
      aj1 = yi - aj1;
      ak += yr;
      ak1 = yi - ak1;
      a[j] = aj;
      a[1 + j] = aj1;
      a[n - j] = ak;
      a[1 + n - j] = ak1;
    }
  else
    {
      xi = a[0] - a[1];
      a[0] += a[1];
      a[1] = xi;
      a[0] *= a[0];
      a[1] *= a[1];
      a[1] = 0.5 * (a[0] - a[1]);
      a[0] -= a[1];
      a[1] = -a[1];
      cc = a[0 + m];
      d = -a[1 + m];
      a[1 + m] = -2.0 * cc * d;
      a[0 + m] = (cc + d) * (cc - d);
      a[1 + m] = -a[1 + m];
    }
}

# define RINT(x) (((x) + A ) - B)

__kernel void normalize_Kernel (__global double *g_x,const int threads,const double A,const double B, __global double *g_err, __global double *g_err2,__global int *g_carry,const __global float *g_inv,const __global double *g_ttp,const __global double *g_ttmp,const __global double *g_ttmpp,double maxerr,const int g_err_flag,__local int * carry)
{
  double temp0, tempErr, ttmpp;
  float inv;

  // read the matrix tile into shared memory
  //unsigned int index = blockIdx.x * threads + threadIdx.x;
  unsigned int threadIdx = get_local_id(0);
  unsigned int index = get_global_id (0);
  if (g_err_flag)
    {
      double terr;
//0
      temp0 = g_x[index];
      temp0 *= g_ttmp[index];
      tempErr = RINT (temp0);
      terr = fabs (temp0 - tempErr);
      if (index)
        temp0 = tempErr;
      else
        temp0 = tempErr - 2.0;
      inv = g_inv[index];
      temp0 *= inv;
      carry[threadIdx + 1] = RINT (temp0);
      temp0 -= carry[threadIdx + 1];
//1    
      ttmpp = g_ttmpp[index];
      temp0 *= ttmpp;
      barrier(CLK_LOCAL_MEM_FENCE); 	
      if (threadIdx)
        temp0 += carry[threadIdx];
      temp0 *= inv;
      carry[threadIdx] = RINT (temp0);
      if (threadIdx == (threads - 1))
        {
          carry[threadIdx + 1] += carry[threadIdx];
          g_carry[index / threads] = carry[threadIdx + 1];
        }
      temp0 -= carry[threadIdx];
//2
      temp0 *= ttmpp;
      barrier(CLK_LOCAL_MEM_FENCE); 	
      if (threadIdx)
        temp0 += carry[threadIdx - 1];
      temp0 *= g_ttp[index];
      g_x[index] = temp0;
      if (terr > maxerr)
        g_err[0] = fmax (terr, g_err[0]);
      g_err2[0] = fmax ((double) fabs(carry[threadIdx]*inv), g_err2[0]);
    }
  else
    {
//0
      temp0 = g_x[index];
      temp0 *= g_ttmp[index];
      temp0 = RINT (temp0);
      if (!index)
        temp0 = temp0 - 2.0;
      inv = g_inv[index];
      temp0 *= inv;
      carry[threadIdx + 1] = RINT (temp0);
      temp0 -= carry[threadIdx + 1];
//1    
      ttmpp = g_ttmpp[index];
      temp0 *= ttmpp;
      barrier(CLK_LOCAL_MEM_FENCE); 	
      if (threadIdx)
        temp0 += carry[threadIdx];
      temp0 *= inv;
      carry[threadIdx] = RINT (temp0);
      if (threadIdx == (threads - 1))
        {
          carry[threadIdx + 1] += carry[threadIdx];
          g_carry[index / threads] = carry[threadIdx + 1];
        }
      temp0 -= carry[threadIdx];
//2
      temp0 *= ttmpp;
      barrier(CLK_LOCAL_MEM_FENCE); 	
      if (threadIdx)
        temp0 += carry[threadIdx - 1];
      temp0 *= g_ttp[index];
      g_x[index] = temp0;
    }
}

__kernel void normalize2_Kernel (__global double *g_x,const int threads,const double A,const double B,__global int *g_carry,const int g_N,const __global float *g_inv2,const __global double *g_ttp2,const __global double *g_ttmp2,const __global float *g_inv3,const __global double *g_ttp3,const __global double *g_ttmp3) 
{
  int threadID = get_global_id (0);
  int j = threads * threadID;
  double temp0, tempErr;
  double temp1, tempErr2;
  int carry;
  if (j < g_N)
    {
      if (threadID)
        carry = g_carry[threadID - 1];
      else
        carry = g_carry[g_N / threads - 1];
      temp0 = g_x[j];
      temp1 = g_x[j + 1];
      tempErr = temp0 * g_ttmp2[threadID];
      tempErr2 = temp1 * g_ttmp3[threadID];
      temp0 = tempErr + carry;
      temp0 *= g_inv2[threadID];
      carry = RINT (temp0);
      temp1 = tempErr2 + carry;
      temp1 *= g_inv3[threadID];
      g_x[j] = (temp0 - carry) * g_ttp2[threadID];
      g_x[j + 1] = temp1 * g_ttp3[threadID];
    }
}
