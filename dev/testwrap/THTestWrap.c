#include "THGeneral.h"
#include "THTestWrap.h"

/* DO NOT RUN
 * void THTestWrap_twodouble(double * a, double * b) {
    *a = 19;
    *b = 83;
} */
void THTestWrap_ReturnOneDouble(double * a) {
    *a = 19;
}

double THTestWrap_CReturnOneDouble() {
    return 19;
}
