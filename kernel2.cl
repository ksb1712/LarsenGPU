#pragma OPENCL EXTENSION cl_khr_fp64 : enable  
#define INF 50000
__kernel void dTrim( __global double *d_number,
                     __global double *d_new1,
                     __global double *d_new,
                      const int M, 
                      const int Z)
{
  int x = get_global_id(0);
  int y = get_global_id(1);
if(y < Z){
    if(x < M - 1){
      d_new1[x * Z + y] = d_number[x * Z + y];
    }
    if(x > 0 && x < M){
      d_new[(x - 1) * Z + y] = d_number[x * Z + y];
    }
  }
}
__kernel
void dProc(__global double *d_new1,__global double *d_new,const int M,const int Z){
int ind = get_global_id(0);
int i;
double  tot1, sum1;
double  tot, sum;
  if(ind < Z){
    tot1 = tot = 0;
    for(i = 0;i < M;i++){
      tot1 += d_new1[i * Z + ind];
      tot += d_new[i * Z + ind];
    }
    tot1 /= M;
    tot /= M;
    sum1 = sum = 0;
    for(i = 0;i < M;i++){
      d_new1[i * Z + ind] -= tot1;
      d_new[i * Z + ind] -= tot;
      sum1 += pow(d_new1[i * Z + ind], 2);
      sum += pow(d_new[i * Z + ind], 2);
    }
    sum1 = sqrt(sum1 / (M - 1));
    sum = sqrt(sum / (M - 1));
    tot1 = tot = 0;
    for(i = 0;i < M;i++){
      d_new1[i * Z + ind] /= sum1;
      d_new[i * Z + ind] /= sum;
      tot1 += pow(d_new1[i * Z + ind], 2);
      tot += pow(d_new[i * Z + ind], 2);
    }
    tot1 = sqrt(tot1);
    tot = sqrt(tot);
    for(i = 0;i < M;i++){
      d_new1[i * Z + ind] /= tot1;
      d_new[i * Z + ind] /= tot;
    }
  }
}
__kernel 
void dInit( __global int *d_nVars, __global int *d_lasso, __global int *d_done, __global int *d_step, const int l)
{
  int ind = get_global_id(0);
 if(ind < l){
    d_nVars[ind] = 0;
    d_lasso[ind] = 0;
    d_done[ind] = 0;
    d_step[ind] = 1;
  }
}    
__kernel
void dCheck(__global int *d_ctrl, __global int *d_step, __global int *d_nVars, const int n, const l, __global int *d_done)
{
  int ind = get_global_id(0);
 if(ind < l && !d_done[ind]){
    if(d_nVars[ind] < n && d_step[ind] < 26){
      d_ctrl[0] = 1;
    }
    else{
      d_done[ind] = 1;
    }
  }
}
__kernel
void dCorr(__global double *d_X, __global double *d_Y, __global double *d_mu,__global double *d_c,__global double *d_, const int M, const int Z,
           const int st, const int l,__global int *d_done)
{
    int ind = get_global_id(0);
    int mod = get_global_id(1);
    if(mod < l && !d_done[mod]){  
    if(ind < Z - 1){
    int i, act, j;
    double  tot;
      i = ind;
      act = st + mod;
      if(i >= act)i++;
      tot = 0;
      for(j = 0;j < M - 1;j++){
        tot += d_X[j * Z + i] * (d_Y[j * Z + act] - d_mu[j * l + mod]);
      }
      d_c[ind * l + mod] = tot;
      d_[ind * l + mod] = tot;
    }
  }
}
__kernel
void dExcCorr(__global double *d_, __global int *d_lVars, __global int *d_nVars, const int l, __global int *d_done)
{
    int ind = get_global_id(0);
    int mod = get_global_id(1);
    if(mod < l && !d_done[mod]){
    if(ind < d_nVars[mod]){
    int i = d_lVars[ind * l + mod];
      d_[i * l + mod] = 0;
    }
  }
}
__kernel
void dMaxcorr(__global double *d_, __global double *d_cmax, __global int *d_ind, const int Z, const int l, __global int *d_done)
{
    int ind = get_global_id(0);
  if(ind < l && !d_done[ind]){
  int j, maxi;
  double  max, tot;
    maxi = -1;
    max = -INF;
    for(j = 0;j < Z - 1;j++){
      tot = fabs(d_[j * l + ind]);
      if(tot > max){
        max = tot;
        maxi = j; 
      }
    }
    d_cmax[ind] = max;
    d_ind[ind] = maxi;
  }
}
__kernel
void dLassoAdd(__global int *d_ind, __global int *d_lVars, __global int *d_nVars,
              __global int *d_lasso, const int l, __global int *d_done)
{
  int ind = get_global_id(0);
  if(ind < l && !d_done[ind]){
    if(!d_lasso[ind]){
      d_lVars[d_nVars[ind] * l + ind] = d_ind[ind];
      d_nVars[ind] += 1;
    }
    else{
      d_lasso[ind] = 0;
    }
  }
}
__kernel
void dXincTY(__global double *d_X, __global double *d_Y, __global double *d_,
            __global int *d_lVars, __global int *d_nVars, const int M, const int Z,
            const int st, const int l, __global int *d_done)
{
  int ind = get_global_id(0);
  int mod = get_global_id(1);
 if(mod < l && !d_done[mod]){
    if(ind < d_nVars[mod]){
    int i, j, act;
    double  tot;
      i = d_lVars[ind * l + mod];
      act = st + mod;
      if(i >= act)i++;
      tot = 0;
      for(j = 0;j < M - 1;j++)tot += d_X[j * Z + i] * d_Y[j * Z + act];
      d_[ind * l + mod] = tot;
    }
  }
}
__kernel
void dSetGram(__global double *d_X, __global double *d_G, __global double *d_I,
             __global int *d_lVars, const int n, const int M, const int Z, const int mod,
              const int st, const int l){
int indx = get_local_id(0);
int indy = get_group_id(0);
 if(indx < n && indy < n && indx <= indy){
  int i, j, k, act;
  double  tot;
    act = mod + st;
    i = d_lVars[indx * l + mod];
    j = d_lVars[indy * l + mod];
    if(i >= act)i++;
    if(j >= act)j++;
    tot = 0;
    for(k = 0;k < M - 1;k++){
      tot += d_X[k * Z + i] * d_X[k * Z + j];
    }
    if(indx == indy){
      d_G[indx * n + indy] = tot;
      d_I[indx * n + indy] = 1;
    }
    else{
      d_G[indx * n + indy] = d_G[indy * n + indx] = tot;
      d_I[indx * n + indy] = d_I[indy * n + indx] = 0;
    }
  }
}
__kernel 
void nodiag_normalize(__global double *A, __global double *I, const int n, const int i){
int y = get_local_id(0);
 if (y < n){
    if (y != i){
      I[i * n + y] /= A[i * n + i];
      A[i * n + y] /= A[i * n + i];
    }
  }
}
__kernel 
void diag_normalize(__global double *A, __global double *I, const int n, const int i){
 I[i * n + i] /= A[i * n + i];
  A[i * n + i] = 1;
}
__kernel
void gaussjordan(__global double *A, __global double *I, const int n, const int i){
int x = get_local_id(0);
int y = get_group_id(0);
  if (x < n && y < n){
    if (x != i){
      I[x * n + y] -= I[i * n + y] * A[x * n + i];
      if(y != i){
        A[x * n + y] -= A[i * n + y] * A[x * n + i];
      }
    }
  }
}

__kernel
void set_zero( __global double *A, __global double *I, const int n, const int i){
int   x = get_local_id(0);
 if (x < n){
    if (x != i){
      A[x * n + i] = 0;
    }
  }
}
__kernel
void dBetaols(__global double *d_I, __global double *d_, __global double *d_betaOLS, const int n, const int mod, const int l){
int ind = get_local_id(0);
  if(ind < n){
  int j;
  double  tot;
    tot = 0;
    for(j = 0;j < n;j++){
      tot += d_I[ind * n + j] * d_[j * l + mod];
    }
    d_betaOLS[ind * l + mod] = tot;
  }
}
__kernel
void ddgamma(__global double *d_X, __global double *d_mu, __global double *d_beta, 
            __global double *d_betaOLS, __global double *d_gamma, __global double *d_d, __global int *d_lVars,
            __global int *d_nVars, const int M, const int Z, const int st, const int l, __global int *d_done){
int ind = get_global_id(0);
int     mod = get_global_id(1);
  if(mod < l && !d_done[mod]){
    if(ind < M - 1){
    int i, j, n, act;
    double  tot;
      n = d_nVars[mod];
      act = st + mod;
      tot = 0;
      for(j = 0;j < n;j++){
        i = d_lVars[j * l + mod];
        if(i >= act)i++;
        tot += d_X[ind * Z + i] * d_betaOLS[j * l + mod];
      }
      d_d[ind * l + mod] = tot - d_mu[ind * l + mod];
      if(ind < n - 1){
        i = d_lVars[ind * l + mod];
                    tot = d_beta[i * l + mod] / (d_beta[i * l + mod] - d_betaOLS[ind * l + mod]);
                  if(tot <= 0)tot = INF;
                    d_gamma[ind * l + mod] = tot;
      }
    }
  }
}
__kernel
void dGammamin(__global double *d_gamma, __global int *d_ind, __global int *d_nVars, const int l, __global int *d_done){
int     ind = get_global_id(0);
  if(ind < l && !d_done[ind]){
  int j, n, mini;
  double  min, tot;
    n = d_nVars[ind];
    min = INF;
    mini = -1;
    tot = 0;
    for(j = 0;j < n - 1;j++){
      tot = d_gamma[j * l + ind];
      if(tot < min){
        min = tot;
        mini = j;
      }
    }
    d_gamma[ind] = min;
    d_ind[ind] = mini;
  }
}
__kernel
void dXTd(__global double *d_X, __global double *d_c, __global double *d_,
        __global double *d_d, __global double *d_cmax, const int M, const int Z,
        const int st,const int l, __global int *d_done){
int     ind = get_global_id(0);
int     mod = get_global_id(1);
  if(mod < l && !d_done[mod]){
    if(ind < Z - 1){
    int i, act, j;
    double  tot, cmax, a, b;
      cmax = d_cmax[mod];
      act = st + mod;
      i = ind;
      if(i >= act)i++;
      tot = 0;
      for(j = 0;j < M - 1;j++){
        tot += d_X[j * Z + i] * d_d[j * l + mod];
      }
      a = (d_c[ind * l + mod] + cmax) / (tot + cmax);
      b = (d_c[ind * l + mod] - cmax) / (tot - cmax);
      if(a <= 0)a = INF;
      if(b <= 0)b = INF;
      tot = min(a, b);
      d_[ind * l + mod] = tot;
    }
  }
}
__kernel
void dExctmp(__global double *d_, __global int *d_lVars, __global int *d_nVars, const int l, __global int *d_done){
int     ind = get_global_id(0);
int     mod = get_global_id(1);
      if(mod < l && !d_done[mod]){
  int i, n;
    n = d_nVars[mod];
                if(ind < n){
      i = d_lVars[ind * l + mod];
      d_[i * l + mod] = INF;
    }
        }
}
__kernel
void dTmpmin(__global double *d_, const int Z, const int l, __global int *d_done){
int     ind = get_global_id(0);
         if(ind < l && !d_done[ind]){
        int     j;
        double   min, tot;
    min = INF;
                for(j = 0;j < Z - 1;j++){
                        tot = d_[j * l + ind];
                        if(tot < min){
                                min = tot;
                        }
                }
                d_[ind] = min;
        }
}
__kernel
void dLassodev(__global double *d_, __global double *d_gamma, __global int *d_nVars,
               __global int *d_lasso, const int n, const int l, __global int *d_done){
int     ind = get_global_id(0);
    if(ind < l && !d_done[ind]){
    if(d_nVars[ind] == n){
      if(d_gamma[ind] < 1){
        d_lasso[ind] = 1;
      }
      else{
        d_gamma[ind] = 1;
      }
    }
    else{
      if(d_gamma[ind] < d_[ind]){
        d_lasso[ind] = 1;
      }
      else{
        d_gamma[ind] = d_[ind];
      }
    }
  }
}
__kernel
void dUpdate( __global double *d_gamma, __global double *d_mu, __global double *d_beta,
            __global double *d_betaOLS, __global double *d_d, __global int *d_lVars,
              __global int *d_nVars, const int M, const int l, __global int *d_done){
int ind = get_global_id(0);
int mod = get_global_id(1);
 if(mod < l && !d_done[mod]){
    if(ind < M - 1){
    int i;
    double  gamma = d_gamma[mod];
      d_mu[ind * l + mod] += gamma * d_d[ind * l + mod];
      if(ind < d_nVars[mod]){
        i = d_lVars[ind * l + mod];
        d_beta[i * l + mod] += gamma * (d_betaOLS[ind * l + mod] - d_beta[i * l + mod]);
      }
    }
  }
}
__kernel
void dLassodrop(__global int *d_ind, __global int *d_lVars, __global int *d_nVars,
              __global int *d_lasso, const int l, __global int *d_done, __global int *val){
int     ind = get_local_id(0);
int     mod = get_group_id(0);
 if(mod < l && !d_done[mod]){
    if(d_lasso[mod]){
    int st, tmp;
      st = d_ind[mod];
      if(ind < d_nVars[mod] - 1 && ind >= st){
        tmp = d_lVars[(ind + 1) * l + mod];
       barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        d_lVars[ind * l + mod] = tmp;
      }
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
       val[ind] = 10;
      if(ind == 0){
        d_nVars[mod] -= 1;

      }
    }
   
      
  }
}
__kernel
void dRess( __global double *d_X, __global double *d_Y, __global double *d_,
          __global double *d_beta, const int M, const int Z, const int st,
          const int l, __global int *d_done){
int ind = get_global_id(0);
int mod = get_global_id(1);
  if(mod < l && !d_done[mod]){
    if(ind < M - 1){
    int i, j, act;
    double  tot;
      act = st + mod;
      tot = 0;
      for(j = 0;j < Z - 1;j++){
        i = j;
        if(i >= act)i++;
        tot += d_X[ind * Z + i] * d_beta[j * l + mod];
      }
      d_[ind * l + mod] = tot - d_Y[ind * Z + act];
    }
  }
}
__kernel
void dFinal( __global double *d_, __global double *d_beta, __global double *d_upper1,
          __global double *d_normb, __global int *d_nVars, __global int *d_step,
          const double g, const int M, const int Z, const int l, __global int *d_done, __global double *lamda){
int     ind = get_global_id(0);
  if(ind < l && !d_done[ind]){
  int i;
  double  upper1 = 0, normb = 0, G = 0;
    for(i = 0;i < Z - 1;i++)
    {
      normb += fabs(d_beta[i * l + ind]);
    }
    for(i = 0;i < M - 1;i++)
    {
      upper1 += pow(d_[i * l + ind], 2);
    }
    upper1 = sqrt(upper1);
    if(d_step[ind] > 1){

         G = (upper1 - d_upper1[ind]) / (d_normb[ind] - normb);
      
      if(G < g)d_done[ind] = 1;
    }
    d_upper1[ind] = upper1;
    d_normb[ind] = normb;
    d_step[ind] += 1;
    lamda[ind] = G;
  
  }
}
