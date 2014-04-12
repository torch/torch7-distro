#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorRandom.c"
#else

TH_API void THTensor_(random)(THTensor *self)
{
  THRandomTLS *rstate = THRandom_getTLS();
#if defined(TH_REAL_IS_BYTE)
  TH_TENSOR_APPLY(real, self, *self_data = (unsigned char)(THRandom_randomWithState(rstate) % (UCHAR_MAX+1)););
#elif defined(TH_REAL_IS_CHAR)
  TH_TENSOR_APPLY(real, self, *self_data = (char)(THRandom_randomWithState(rstate) % (CHAR_MAX+1)););
#elif defined(TH_REAL_IS_SHORT)
  TH_TENSOR_APPLY(real, self, *self_data = (short)(THRandom_randomWithState(rstate) % (SHRT_MAX+1)););
#elif defined(TH_REAL_IS_INT)
  TH_TENSOR_APPLY(real, self, *self_data = (int)(THRandom_randomWithState(rstate) % (INT_MAX+1UL)););
#elif defined(TH_REAL_IS_LONG)
  TH_TENSOR_APPLY(real, self, *self_data = (long)(THRandom_randomWithState(rstate) % (LONG_MAX+1UL)););
#elif defined(TH_REAL_IS_FLOAT)
  TH_TENSOR_APPLY(real, self, *self_data = (float)(THRandom_randomWithState(rstate) % ((1UL << FLT_MANT_DIG)+1)););
#elif defined(TH_REAL_IS_DOUBLE)
  TH_TENSOR_APPLY(real, self, *self_data = (float)(THRandom_randomWithState(rstate) % ((1UL << DBL_MANT_DIG)+1)););
#else
#error "Unknown type"
#endif
}

TH_API void THTensor_(geometric)(THTensor *self, double p)
{
  THRandomTLS *rstate = THRandom_getTLS();
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_geometricWithState(rstate, p););
}

TH_API void THTensor_(bernoulli)(THTensor *self, double p)
{
  THRandomTLS *rstate = THRandom_getTLS();
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_bernoulliWithState(rstate, p););
}

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

TH_API void THTensor_(uniform)(THTensor *self, double a, double b)
{
  THRandomTLS *rstate = THRandom_getTLS();
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_uniformWithState(rstate, a, b););
}

TH_API void THTensor_(normal)(THTensor *self, double mean, double stdv)
{
  THRandomTLS *rstate = THRandom_getTLS();
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_normalWithState(rstate, mean, stdv););
}

TH_API void THTensor_(exponential)(THTensor *self, double lambda)
{
  THRandomTLS *rstate = THRandom_getTLS();
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_exponentialWithState(rstate, lambda););
}

TH_API void THTensor_(cauchy)(THTensor *self, double median, double sigma)
{
  THRandomTLS *rstate = THRandom_getTLS();
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_cauchyWithState(rstate, median, sigma););
}

TH_API void THTensor_(logNormal)(THTensor *self, double mean, double stdv)
{
  THRandomTLS *rstate = THRandom_getTLS();
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_logNormalWithState(rstate, mean, stdv););
}

#endif

#if defined(TH_REAL_IS_LONG)
TH_API void THTensor_(getRNGState)(THTensor *self)
{
  unsigned long *data;
  long *offset;
  long *left;

  THTensor_(resize1d)(self,626);
  data = (unsigned long *)THTensor_(data)(self);
  offset = (long *)data+624;
  left = (long *)data+625;

  THRandom_getState(data,offset,left);
}

TH_API void THTensor_(setRNGState)(THTensor *self)
{
  unsigned long *data;
  long *offset;
  long *left;

  THArgCheck(THTensor_(nElement)(self) == 626, 1, "state should have 626 elements");
  data = (unsigned long *)THTensor_(data)(self);
  offset = (long *)(data+624);
  left = (long *)(data+625);

  THRandom_setState(data,*offset,*left);
}

#endif

#endif
