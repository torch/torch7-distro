#ifndef TH_RANDOM_INC
#define TH_RANDOM_INC

#include "THRandomNumber.h"

#define THTensor          TH_CONCAT_3(TH,Real,Tensor)
#define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)

/* random numbers */
#include "generic/THTensorRandom.h"
#include "THGenerateAllTypes.h"

#endif
