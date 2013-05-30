#include "THGeneral.h"
#include "THTestWrap.h"

void THTestWrap_ReturnOneDouble(double * a) {
    *a = 19;
}

double THTestWrap_CReturnOneDouble() {
    return 19;
}

void THTestWrap_ReturnOneIndex(long * a) {
    *a = 19;
}

void THTestWrap_ReturnOneBoolean(int * a) {
    *a = 1;
}

