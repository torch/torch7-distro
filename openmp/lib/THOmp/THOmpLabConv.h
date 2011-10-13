#include "TH.h"

#ifdef _WIN32
# ifdef THOmp_EXPORTS
#  define THOMP_API __declspec(dllexport)
# else
#  define THOMP_API __declspec(dllimport)
# endif
#else
# define THOMP_API /**/
#endif

#define THOmpLab_(NAME)   TH_CONCAT_4(THOmp,Real,Lab_,NAME)

#include "generic/THOmpLabConv.h"
#include "THGenerateAllTypes.h"

