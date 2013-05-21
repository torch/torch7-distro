#include "THGeneral.h"
#include "THTestWrap.h"

/* DO NOT RUN
 * void THTestWrap_twodouble(double * a, double * b) {
    *a = 19;
    *b = 83;
} */
void THTestWrap_onedouble(double * a) {
    *a = 19;
}

double THTestWrap_onedoublecreturned() {
    return 19;
}
