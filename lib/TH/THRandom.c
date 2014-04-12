#include "THGeneral.h"
#include "THRandom.h"

/* Code for the Mersenne Twister random generator.... */
#define n 624
#define m 397

#ifdef C_HAS_TLS_POSIX
#define TLSPREFIX __thread
#elif C_HAS_TLS_WIN32
#define TLSPREFIX __declspec( thread )
#else
#define TLSPREFIX
#endif

#if defined(C_HAS_PTHREADS)
#include <pthread.h>
static pthread_once_t THRandom_tlsInitFlag = PTHREAD_ONCE_INIT;
static pthread_key_t THRandom_defaultTLSkey;
#else
static TLSPREFIX THRandomTLS* THRandom_defaultTLS = NULL;
#endif

void THRandom_initializeTLS()
{  
  THRandomTLS* state = (THRandomTLS*) malloc(sizeof(THRandomTLS));
  (*state).left = 1;
  (*state).initf = 0;
  (*state).normal_is_valid = 0;

#if defined(C_HAS_PTHREADS)
  if (pthread_setspecific(THRandom_defaultTLSkey, (void *)state) != 0) {
    THError("pthread could not generate any more keys in pthread_setspecific");
  }
#else
  THRandom_defaultTLS = state;
#endif
}

#if defined(C_HAS_PTHREADS)
static void destroy_tls() {
  THRandomTLS* rstate = (THRandomTLS*)pthread_getspecific(THRandom_defaultTLSkey);
  if (rstate != NULL) free(rstate);  
}
static void create_tls_key() {
  if (pthread_key_create(&THRandom_defaultTLSkey, destroy_tls)) {
    THError("pthread could not generate any more keys in pthread_key_create");
  }
}
#endif

THRandomTLS* THRandom_getTLS()
{
#if defined(C_HAS_PTHREADS)
  pthread_once(&THRandom_tlsInitFlag, create_tls_key);
  THRandomTLS* rstate = (THRandomTLS*)pthread_getspecific(THRandom_defaultTLSkey);
  if (rstate == NULL) {
    THRandom_initializeTLS();
    rstate = (THRandomTLS*)pthread_getspecific(THRandom_defaultTLSkey);
  }
  return rstate;
#else
  if (THRandom_defaultTLS == NULL) THRandom_initializeTLS();
  return THRandom_defaultTLS;
#endif
}

unsigned long THRandom_seed()
{
  unsigned long s = (unsigned long)time(0);
  THRandom_manualSeed(s);
  return s;
}

/* The next 4 methods are taken from http:www.math.keio.ac.jpmatumotoemt.html
   Here is the copyright:
   Some minor modifications have been made to adapt to "my" C... */

/*
   A C-program for MT19937, with initialization improved 2002/2/10.
   Coded by Takuji Nishimura and Makoto Matsumoto.
   This is a faster version by taking Shawn Cokus's optimization,
   Matthe Bellew's simplification, Isaku Wada's double version.

   Before using, initialize the state by using init_genrand(seed)
   or init_by_array(init_key, key_length).

   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote
        products derived from this software without specific prior written
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


   Any feedback is very welcome.
   http://www.math.keio.ac.jp/matumoto/emt.html
   email: matumoto@math.keio.ac.jp
*/

/* Macros for the Mersenne Twister random generator... */
/* Period parameters */  
/* #define n 624 */
/* #define m 397 */
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UMASK 0x80000000UL /* most significant w-r bits */
#define LMASK 0x7fffffffUL /* least significant r bits */
#define MIXBITS(u,v) ( ((u) & UMASK) | ((v) & LMASK) )
#define TWIST(u,v) ((MIXBITS(u,v) >> 1) ^ ((v)&1UL ? MATRIX_A : 0UL))
/*********************************************************** That's it. */

void THRandom_manualSeed(unsigned long the_seed_)
{
  THRandomTLS *rstate = THRandom_getTLS();
  int j;
  (*rstate).the_initial_seed = the_seed_;
  (*rstate).state[0]= (*rstate).the_initial_seed & 0xffffffffUL;
  for(j = 1; j < n; j++)
  {
    (*rstate).state[j] = (1812433253UL * ((*rstate).state[j-1] ^ ((*rstate).state[j-1] >> 30)) + j); 
    /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
    /* In the previous versions, mSBs of the seed affect   */
    /* only mSBs of the array state[].                        */
    /* 2002/01/09 modified by makoto matsumoto             */
    (*rstate).state[j] &= 0xffffffffUL;  /* for >32 bit machines */
  }
  (*rstate).left = 1;
  (*rstate).initf = 1;
}

unsigned long THRandom_initialSeed()
{
  THRandomTLS *rstate = THRandom_getTLS();
  if((*rstate).initf == 0)
  {
    THRandom_seed();
  }

  return (*rstate).the_initial_seed;
}

void THRandom_nextState(THRandomTLS* rstate)
{
  unsigned long *p=(*rstate).state;
  int j;

  /* if init_genrand() has not been called, */
  /* a default initial seed is used         */
  if((*rstate).initf == 0)
    THRandom_seed();

  (*rstate).left = n;
  (*rstate).next = (*rstate).state;
    
  for(j = n-m+1; --j; p++) 
    *p = p[m] ^ TWIST(p[0], p[1]);

  for(j = m; --j; p++) 
    *p = p[m-n] ^ TWIST(p[0], p[1]);

  *p = p[m-n] ^ TWIST(p[0], (*rstate).state[0]);
}

unsigned long THRandom_random()
{
  THRandomTLS *rstate = THRandom_getTLS();
  return THRandom_randomWithState(rstate);
}

unsigned long THRandom_randomWithState(THRandomTLS* rstate)
{
  unsigned long y;

  if (--(*rstate).left == 0)
    THRandom_nextState(rstate);
  y = *((*rstate).next)++;
  
  /* Tempering */
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680UL;
  y ^= (y << 15) & 0xefc60000UL;
  y ^= (y >> 18);

  return y;
}

/* generates a random number on [0,1)-double-interval */
static double __uniform__(THRandomTLS* rstate)
{
  unsigned long y;
  if(--((*rstate).left) == 0)
    THRandom_nextState(rstate);
  y = *((*rstate).next)++;

  /* Tempering */
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680UL;
  y ^= (y << 15) & 0xefc60000UL;
  y ^= (y >> 18);
  
  return (double)y * (1.0/4294967296.0); 
  /* divided by 2^32 */
}

/*********************************************************

 Thanks *a lot* Takuji Nishimura and Makoto Matsumoto!

 Now my own code...

*********************************************************/
double THRandom_uniform(double a, double b)
{
  THRandomTLS *rstate = THRandom_getTLS();
  return THRandom_uniformWithState(rstate, a, b);
}
double THRandom_uniformWithState(THRandomTLS* rstate, double a, double b)
{
  return(__uniform__(rstate) * (b - a) + a);
}

double THRandom_normal(double mean, double stdv)
{
  THRandomTLS *rstate = THRandom_getTLS();
  return THRandom_normalWithState(rstate, mean, stdv);
}

double THRandom_normalWithState(THRandomTLS* rstate, double mean, double stdv)
{
  THArgCheck(stdv > 0, 2, "standard deviation must be strictly positive");

  if(!(*rstate).normal_is_valid)
  {
    (*rstate).normal_x = __uniform__(rstate);
    (*rstate).normal_y = __uniform__(rstate);
    (*rstate).normal_rho = sqrt(-2. * log(1.0-(*rstate).normal_y));
    (*rstate).normal_is_valid = 1;
  }
  else
    (*rstate).normal_is_valid = 0;
  
  if((*rstate).normal_is_valid)
    return (*rstate).normal_rho*cos(2.*M_PI*(*rstate).normal_x)*stdv+mean;
  else
    return (*rstate).normal_rho*sin(2.*M_PI*(*rstate).normal_x)*stdv+mean;
}

double THRandom_exponential(double lambda)
{
  THRandomTLS *rstate = THRandom_getTLS();
  return THRandom_exponentialWithState(rstate, lambda);
}
 
double THRandom_exponentialWithState(THRandomTLS* rstate, double lambda)
{
  return(-1. / lambda * log(1-__uniform__(rstate)));
}

double THRandom_cauchy(double median, double sigma)
{
  THRandomTLS *rstate = THRandom_getTLS();
  return THRandom_cauchyWithState(rstate, median, sigma);
}
 
double THRandom_cauchyWithState(THRandomTLS* rstate, double median, double sigma)
{
  return(median + sigma * tan(M_PI*(__uniform__(rstate)-0.5)));
}

/* Faut etre malade pour utiliser ca.
   M'enfin. */
double THRandom_logNormal(double mean, double stdv)
{
  THRandomTLS *rstate = THRandom_getTLS();
  return THRandom_logNormalWithState(rstate, mean, stdv);
}
double THRandom_logNormalWithState(THRandomTLS* rstate, double mean, double stdv)
{
  double zm = mean*mean;
  double zs = stdv*stdv;
  THArgCheck(stdv > 0, 2, "standard deviation must be strictly positive");
  return(exp(THRandom_normalWithState(rstate, log(zm/sqrt(zs + zm)), sqrt(log(zs/zm+1)) )));
}

int THRandom_geometric(double p)
{
  THRandomTLS *rstate = THRandom_getTLS();
  return THRandom_geometricWithState(rstate, p);
}

int THRandom_geometricWithState(THRandomTLS* rstate, double p)
{
  THArgCheck(p > 0 && p < 1, 1, "must be > 0 and < 1");
  return((int)(log(1-__uniform__(rstate)) / log(p)) + 1);
}

int THRandom_bernoulli(double p)
{
  THRandomTLS *rstate = THRandom_getTLS();
  return THRandom_bernoulliWithState(rstate, p);
}

int THRandom_bernoulliWithState(THRandomTLS* rstate, double p)
{
  THArgCheck(p >= 0 && p <= 1, 1, "must be >= 0 and <= 1");
  return(__uniform__(rstate) <= p);
}

/* returns the random number state */
void THRandom_getState(unsigned long *_state, long *offset, long *_left)
{
  THRandomTLS *rstate = THRandom_getTLS();
  if((*rstate).initf == 0)
    THRandom_seed();
  memmove(_state, (*rstate).state, n*sizeof(long));
  *offset = (long)((*rstate).next - (*rstate).state);
  *_left = (*rstate).left;
}

/* sets the random number state */
void THRandom_setState(unsigned long *_state, long offset, long _left)
{
  THRandomTLS *rstate = THRandom_getTLS();
  memmove((*rstate).state, _state, n*sizeof(long));
  (*rstate).next = (*rstate).state + offset;
  (*rstate).left = _left;
  (*rstate).initf = 1;
}
