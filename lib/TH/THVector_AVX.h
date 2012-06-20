
void THFloatVector_conv1dk3(float * y, float * x, float * c, float alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 2;
float *yptr, *xptr, *cptr;
__m256 ymm0, ymm1, ymm2, ymm6, ymm8, ymm9, ymm10, ymm11, ymm13, ymm14, ymm15;
float ctmp[3];
for(j = 0; j < 3; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-8) && i <= (xn - 16); i+=8){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_ps(yptr);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm13 = _mm256_loadu_ps(xptr);
ymm15 = _mm256_loadu_ps(xptr + 8);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_storeu_ps(yptr, ymm8);}
for(; i < n; i+=8){
int xmask[16] = {0};
int ymask[8] = {0};
int m;
for(m = 0; (m + i < n) && (m < 8); m++){
ymask[m] = 0xFFFFFFFF;
xmask[m] = 0xFFFFFFFF;}
for(; (m + i) < xn && (m < 16); m++){
xmask[m] = 0xFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi32 (ymask[7], ymask[6], ymask[5], ymask[4], ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_ps (yptr, ymm12);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
__m256i ymm3 =  _mm256_set_epi32 (xmask[7], xmask[6], xmask[5], xmask[4], xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm13 =  _mm256_maskload_ps (xptr, ymm3);
ymm3 =  _mm256_set_epi32 (xmask[15],xmask[14],xmask[13],xmask[12],xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm15 =  _mm256_maskload_ps (xptr + 8, ymm3);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_maskstore_ps (yptr, ymm12, ymm8);}
}

void THFloatVector_conv1dk4(float * y, float * x, float * c, float alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 3;
float *yptr, *xptr, *cptr;
__m256 ymm0, ymm1, ymm2, ymm6, ymm8, ymm9, ymm10, ymm11, ymm13, ymm14, ymm15;
float ctmp[4];
for(j = 0; j < 4; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-8) && i <= (xn - 16); i+=8){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_ps(yptr);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
ymm13 = _mm256_loadu_ps(xptr);
ymm15 = _mm256_loadu_ps(xptr + 8);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_storeu_ps(yptr, ymm8);}
for(; i < n; i+=8){
int xmask[16] = {0};
int ymask[8] = {0};
int m;
for(m = 0; (m + i < n) && (m < 8); m++){
ymask[m] = 0xFFFFFFFF;
xmask[m] = 0xFFFFFFFF;}
for(; (m + i) < xn && (m < 16); m++){
xmask[m] = 0xFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi32 (ymask[7], ymask[6], ymask[5], ymask[4], ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_ps (yptr, ymm12);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
__m256i ymm3 =  _mm256_set_epi32 (xmask[7], xmask[6], xmask[5], xmask[4], xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm13 =  _mm256_maskload_ps (xptr, ymm3);
ymm3 =  _mm256_set_epi32 (xmask[15],xmask[14],xmask[13],xmask[12],xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm15 =  _mm256_maskload_ps (xptr + 8, ymm3);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_maskstore_ps (yptr, ymm12, ymm8);}
}

void THFloatVector_conv1dk5(float * y, float * x, float * c, float alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 4;
float *yptr, *xptr, *cptr;
__m256 ymm0, ymm1, ymm2, ymm6, ymm8, ymm9, ymm10, ymm11, ymm13, ymm14, ymm15;
float ctmp[5];
for(j = 0; j < 5; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-8) && i <= (xn - 16); i+=8){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_ps(yptr);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
ymm13 = _mm256_loadu_ps(xptr);
ymm15 = _mm256_loadu_ps(xptr + 8);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_storeu_ps(yptr, ymm8);}
for(; i < n; i+=8){
int xmask[16] = {0};
int ymask[8] = {0};
int m;
for(m = 0; (m + i < n) && (m < 8); m++){
ymask[m] = 0xFFFFFFFF;
xmask[m] = 0xFFFFFFFF;}
for(; (m + i) < xn && (m < 16); m++){
xmask[m] = 0xFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi32 (ymask[7], ymask[6], ymask[5], ymask[4], ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_ps (yptr, ymm12);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
__m256i ymm3 =  _mm256_set_epi32 (xmask[7], xmask[6], xmask[5], xmask[4], xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm13 =  _mm256_maskload_ps (xptr, ymm3);
ymm3 =  _mm256_set_epi32 (xmask[15],xmask[14],xmask[13],xmask[12],xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm15 =  _mm256_maskload_ps (xptr + 8, ymm3);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_maskstore_ps (yptr, ymm12, ymm8);}
}

void THFloatVector_conv1dk6(float * y, float * x, float * c, float alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 5;
float *yptr, *xptr, *cptr;
__m256 ymm0, ymm1, ymm2, ymm6, ymm8, ymm9, ymm10, ymm11, ymm13, ymm14, ymm15;
float ctmp[6];
for(j = 0; j < 6; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-8) && i <= (xn - 16); i+=8){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_ps(yptr);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
ymm13 = _mm256_loadu_ps(xptr);
ymm15 = _mm256_loadu_ps(xptr + 8);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 5);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_storeu_ps(yptr, ymm8);}
for(; i < n; i+=8){
int xmask[16] = {0};
int ymask[8] = {0};
int m;
for(m = 0; (m + i < n) && (m < 8); m++){
ymask[m] = 0xFFFFFFFF;
xmask[m] = 0xFFFFFFFF;}
for(; (m + i) < xn && (m < 16); m++){
xmask[m] = 0xFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi32 (ymask[7], ymask[6], ymask[5], ymask[4], ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_ps (yptr, ymm12);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
__m256i ymm3 =  _mm256_set_epi32 (xmask[7], xmask[6], xmask[5], xmask[4], xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm13 =  _mm256_maskload_ps (xptr, ymm3);
ymm3 =  _mm256_set_epi32 (xmask[15],xmask[14],xmask[13],xmask[12],xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm15 =  _mm256_maskload_ps (xptr + 8, ymm3);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 5);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_maskstore_ps (yptr, ymm12, ymm8);}
}

void THFloatVector_conv1dk7(float * y, float * x, float * c, float alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 6;
float *yptr, *xptr, *cptr;
__m256 ymm0, ymm1, ymm2, ymm6, ymm8, ymm9, ymm10, ymm11, ymm13, ymm14, ymm15;
float ctmp[7];
for(j = 0; j < 7; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-8) && i <= (xn - 16); i+=8){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_ps(yptr);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
ymm13 = _mm256_loadu_ps(xptr);
ymm15 = _mm256_loadu_ps(xptr + 8);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 5);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 6);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_storeu_ps(yptr, ymm8);}
for(; i < n; i+=8){
int xmask[16] = {0};
int ymask[8] = {0};
int m;
for(m = 0; (m + i < n) && (m < 8); m++){
ymask[m] = 0xFFFFFFFF;
xmask[m] = 0xFFFFFFFF;}
for(; (m + i) < xn && (m < 16); m++){
xmask[m] = 0xFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi32 (ymask[7], ymask[6], ymask[5], ymask[4], ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_ps (yptr, ymm12);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
__m256i ymm3 =  _mm256_set_epi32 (xmask[7], xmask[6], xmask[5], xmask[4], xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm13 =  _mm256_maskload_ps (xptr, ymm3);
ymm3 =  _mm256_set_epi32 (xmask[15],xmask[14],xmask[13],xmask[12],xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm15 =  _mm256_maskload_ps (xptr + 8, ymm3);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 5);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 6);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_maskstore_ps (yptr, ymm12, ymm8);}
}

void THFloatVector_conv1dk8(float * y, float * x, float * c, float alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 7;
float *yptr, *xptr, *cptr;
__m256 ymm0, ymm1, ymm2, ymm6, ymm8, ymm9, ymm10, ymm11, ymm13, ymm14, ymm15;
float ctmp[8];
for(j = 0; j < 8; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-8) && i <= (xn - 16); i+=8){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_ps(yptr);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
ymm13 = _mm256_loadu_ps(xptr);
ymm15 = _mm256_loadu_ps(xptr + 8);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm2 = _mm256_shuffle_ps(ymm1,  ymm15, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 5);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 6);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 7);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_storeu_ps(yptr, ymm8);}
for(; i < n; i+=8){
int xmask[16] = {0};
int ymask[8] = {0};
int m;
for(m = 0; (m + i < n) && (m < 8); m++){
ymask[m] = 0xFFFFFFFF;
xmask[m] = 0xFFFFFFFF;}
for(; (m + i) < xn && (m < 16); m++){
xmask[m] = 0xFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi32 (ymask[7], ymask[6], ymask[5], ymask[4], ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_ps (yptr, ymm12);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
__m256i ymm3 =  _mm256_set_epi32 (xmask[7], xmask[6], xmask[5], xmask[4], xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm13 =  _mm256_maskload_ps (xptr, ymm3);
ymm3 =  _mm256_set_epi32 (xmask[15],xmask[14],xmask[13],xmask[12],xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm15 =  _mm256_maskload_ps (xptr + 8, ymm3);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm2 = _mm256_shuffle_ps(ymm1,  ymm15, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 5);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 6);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 7);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_maskstore_ps (yptr, ymm12, ymm8);}
}

void THFloatVector_conv1dk9(float * y, float * x, float * c, float alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 8;
float *yptr, *xptr, *cptr;
__m256 ymm0, ymm1, ymm2, ymm6, ymm8, ymm9, ymm10, ymm11, ymm13, ymm14, ymm15;
float ctmp[9];
for(j = 0; j < 9; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-8) && i <= (xn - 16); i+=8){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_ps(yptr);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
ymm13 = _mm256_loadu_ps(xptr);
ymm15 = _mm256_loadu_ps(xptr + 8);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm2 = _mm256_shuffle_ps(ymm1,  ymm15, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 5);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 6);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 7);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm13 = ymm15;
ymm6 = _mm256_broadcast_ss(cptr + 8);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_storeu_ps(yptr, ymm8);}
for(; i < n; i+=8){
int xmask[16] = {0};
int ymask[8] = {0};
int m;
for(m = 0; (m + i < n) && (m < 8); m++){
ymask[m] = 0xFFFFFFFF;
xmask[m] = 0xFFFFFFFF;}
for(; (m + i) < xn && (m < 16); m++){
xmask[m] = 0xFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi32 (ymask[7], ymask[6], ymask[5], ymask[4], ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_ps (yptr, ymm12);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
__m256i ymm3 =  _mm256_set_epi32 (xmask[7], xmask[6], xmask[5], xmask[4], xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm13 =  _mm256_maskload_ps (xptr, ymm3);
ymm3 =  _mm256_set_epi32 (xmask[15],xmask[14],xmask[13],xmask[12],xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm15 =  _mm256_maskload_ps (xptr + 8, ymm3);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm2 = _mm256_shuffle_ps(ymm1,  ymm15, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 5);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 6);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 7);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm13 = ymm15;
ymm6 = _mm256_broadcast_ss(cptr + 8);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_maskstore_ps (yptr, ymm12, ymm8);}
}

void THFloatVector_conv1dk10(float * y, float * x, float * c, float alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 9;
float *yptr, *xptr, *cptr;
__m256 ymm0, ymm1, ymm2, ymm6, ymm8, ymm9, ymm10, ymm11, ymm13, ymm14, ymm15;
float ctmp[10];
for(j = 0; j < 10; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-8) && i <= (xn - 24); i+=8){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_ps(yptr);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
ymm13 = _mm256_loadu_ps(xptr);
ymm15 = _mm256_loadu_ps(xptr + 8);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm2 = _mm256_shuffle_ps(ymm1,  ymm15, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 5);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 6);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 7);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm13 = ymm15;
ymm15 = _mm256_loadu_ps(xptr + 16);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 8);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 9);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_storeu_ps(yptr, ymm8);}
for(; i < n; i+=8){
int xmask[24] = {0};
int ymask[8] = {0};
int m;
for(m = 0; (m + i < n) && (m < 8); m++){
ymask[m] = 0xFFFFFFFF;
xmask[m] = 0xFFFFFFFF;}
for(; (m + i) < xn && (m < 24); m++){
xmask[m] = 0xFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi32 (ymask[7], ymask[6], ymask[5], ymask[4], ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_ps (yptr, ymm12);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
__m256i ymm3 =  _mm256_set_epi32 (xmask[7], xmask[6], xmask[5], xmask[4], xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm13 =  _mm256_maskload_ps (xptr, ymm3);
ymm3 =  _mm256_set_epi32 (xmask[15],xmask[14],xmask[13],xmask[12],xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm15 =  _mm256_maskload_ps (xptr + 8, ymm3);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm2 = _mm256_shuffle_ps(ymm1,  ymm15, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 5);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 6);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 7);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm13 = ymm15;
ymm3 =  _mm256_set_epi32 (xmask[23],xmask[22],xmask[21],xmask[20],xmask[19],xmask[18],xmask[17],xmask[16]); 
ymm15 =  _mm256_maskload_ps (xptr + 16, ymm3);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 8);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 9);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_maskstore_ps (yptr, ymm12, ymm8);}
}

void THFloatVector_conv1dk11(float * y, float * x, float * c, float alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 10;
float *yptr, *xptr, *cptr;
__m256 ymm0, ymm1, ymm2, ymm6, ymm8, ymm9, ymm10, ymm11, ymm13, ymm14, ymm15;
float ctmp[11];
for(j = 0; j < 11; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-8) && i <= (xn - 24); i+=8){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_ps(yptr);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
ymm13 = _mm256_loadu_ps(xptr);
ymm15 = _mm256_loadu_ps(xptr + 8);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm2 = _mm256_shuffle_ps(ymm1,  ymm15, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 5);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 6);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 7);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm13 = ymm15;
ymm15 = _mm256_loadu_ps(xptr + 16);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 8);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 9);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 10);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_storeu_ps(yptr, ymm8);}
for(; i < n; i+=8){
int xmask[24] = {0};
int ymask[8] = {0};
int m;
for(m = 0; (m + i < n) && (m < 8); m++){
ymask[m] = 0xFFFFFFFF;
xmask[m] = 0xFFFFFFFF;}
for(; (m + i) < xn && (m < 24); m++){
xmask[m] = 0xFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi32 (ymask[7], ymask[6], ymask[5], ymask[4], ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_ps (yptr, ymm12);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
__m256i ymm3 =  _mm256_set_epi32 (xmask[7], xmask[6], xmask[5], xmask[4], xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm13 =  _mm256_maskload_ps (xptr, ymm3);
ymm3 =  _mm256_set_epi32 (xmask[15],xmask[14],xmask[13],xmask[12],xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm15 =  _mm256_maskload_ps (xptr + 8, ymm3);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm2 = _mm256_shuffle_ps(ymm1,  ymm15, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 5);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 6);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 7);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm13 = ymm15;
ymm3 =  _mm256_set_epi32 (xmask[23],xmask[22],xmask[21],xmask[20],xmask[19],xmask[18],xmask[17],xmask[16]); 
ymm15 =  _mm256_maskload_ps (xptr + 16, ymm3);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 8);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 9);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 10);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_maskstore_ps (yptr, ymm12, ymm8);}
}

void THFloatVector_conv1dk12(float * y, float * x, float * c, float alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 11;
float *yptr, *xptr, *cptr;
__m256 ymm0, ymm1, ymm2, ymm6, ymm8, ymm9, ymm10, ymm11, ymm13, ymm14, ymm15;
float ctmp[12];
for(j = 0; j < 12; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-8) && i <= (xn - 24); i+=8){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_ps(yptr);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
ymm13 = _mm256_loadu_ps(xptr);
ymm15 = _mm256_loadu_ps(xptr + 8);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm2 = _mm256_shuffle_ps(ymm1,  ymm15, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 5);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 6);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 7);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm13 = ymm15;
ymm15 = _mm256_loadu_ps(xptr + 16);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 8);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 9);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 10);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 11);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_storeu_ps(yptr, ymm8);}
for(; i < n; i+=8){
int xmask[24] = {0};
int ymask[8] = {0};
int m;
for(m = 0; (m + i < n) && (m < 8); m++){
ymask[m] = 0xFFFFFFFF;
xmask[m] = 0xFFFFFFFF;}
for(; (m + i) < xn && (m < 24); m++){
xmask[m] = 0xFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi32 (ymask[7], ymask[6], ymask[5], ymask[4], ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_ps (yptr, ymm12);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
__m256i ymm3 =  _mm256_set_epi32 (xmask[7], xmask[6], xmask[5], xmask[4], xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm13 =  _mm256_maskload_ps (xptr, ymm3);
ymm3 =  _mm256_set_epi32 (xmask[15],xmask[14],xmask[13],xmask[12],xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm15 =  _mm256_maskload_ps (xptr + 8, ymm3);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm2 = _mm256_shuffle_ps(ymm1,  ymm15, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 5);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 6);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 7);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm13 = ymm15;
ymm3 =  _mm256_set_epi32 (xmask[23],xmask[22],xmask[21],xmask[20],xmask[19],xmask[18],xmask[17],xmask[16]); 
ymm15 =  _mm256_maskload_ps (xptr + 16, ymm3);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 8);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 9);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 10);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 11);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_maskstore_ps (yptr, ymm12, ymm8);}
}

void THFloatVector_conv1dk13(float * y, float * x, float * c, float alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 12;
float *yptr, *xptr, *cptr;
__m256 ymm0, ymm1, ymm2, ymm6, ymm8, ymm9, ymm10, ymm11, ymm13, ymm14, ymm15;
float ctmp[13];
for(j = 0; j < 13; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-8) && i <= (xn - 24); i+=8){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_ps(yptr);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
ymm13 = _mm256_loadu_ps(xptr);
ymm15 = _mm256_loadu_ps(xptr + 8);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm2 = _mm256_shuffle_ps(ymm1,  ymm15, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 5);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 6);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 7);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm13 = ymm15;
ymm15 = _mm256_loadu_ps(xptr + 16);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 8);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 9);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 10);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 11);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 12);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_storeu_ps(yptr, ymm8);}
for(; i < n; i+=8){
int xmask[24] = {0};
int ymask[8] = {0};
int m;
for(m = 0; (m + i < n) && (m < 8); m++){
ymask[m] = 0xFFFFFFFF;
xmask[m] = 0xFFFFFFFF;}
for(; (m + i) < xn && (m < 24); m++){
xmask[m] = 0xFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi32 (ymask[7], ymask[6], ymask[5], ymask[4], ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_ps (yptr, ymm12);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
__m256i ymm3 =  _mm256_set_epi32 (xmask[7], xmask[6], xmask[5], xmask[4], xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm13 =  _mm256_maskload_ps (xptr, ymm3);
ymm3 =  _mm256_set_epi32 (xmask[15],xmask[14],xmask[13],xmask[12],xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm15 =  _mm256_maskload_ps (xptr + 8, ymm3);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm2 = _mm256_shuffle_ps(ymm1,  ymm15, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 5);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 6);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 7);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm13 = ymm15;
ymm3 =  _mm256_set_epi32 (xmask[23],xmask[22],xmask[21],xmask[20],xmask[19],xmask[18],xmask[17],xmask[16]); 
ymm15 =  _mm256_maskload_ps (xptr + 16, ymm3);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 8);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 9);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 10);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 11);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 12);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_maskstore_ps (yptr, ymm12, ymm8);}
}

void THFloatVector_conv1dk14(float * y, float * x, float * c, float alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 13;
float *yptr, *xptr, *cptr;
__m256 ymm0, ymm1, ymm2, ymm6, ymm8, ymm9, ymm10, ymm11, ymm13, ymm14, ymm15;
float ctmp[14];
for(j = 0; j < 14; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-8) && i <= (xn - 24); i+=8){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_ps(yptr);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
ymm13 = _mm256_loadu_ps(xptr);
ymm15 = _mm256_loadu_ps(xptr + 8);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm2 = _mm256_shuffle_ps(ymm1,  ymm15, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 5);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 6);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 7);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm13 = ymm15;
ymm15 = _mm256_loadu_ps(xptr + 16);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 8);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 9);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 10);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 11);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm6 = _mm256_broadcast_ss(cptr + 12);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 13);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_storeu_ps(yptr, ymm8);}
for(; i < n; i+=8){
int xmask[24] = {0};
int ymask[8] = {0};
int m;
for(m = 0; (m + i < n) && (m < 8); m++){
ymask[m] = 0xFFFFFFFF;
xmask[m] = 0xFFFFFFFF;}
for(; (m + i) < xn && (m < 24); m++){
xmask[m] = 0xFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi32 (ymask[7], ymask[6], ymask[5], ymask[4], ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_ps (yptr, ymm12);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
__m256i ymm3 =  _mm256_set_epi32 (xmask[7], xmask[6], xmask[5], xmask[4], xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm13 =  _mm256_maskload_ps (xptr, ymm3);
ymm3 =  _mm256_set_epi32 (xmask[15],xmask[14],xmask[13],xmask[12],xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm15 =  _mm256_maskload_ps (xptr + 8, ymm3);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm2 = _mm256_shuffle_ps(ymm1,  ymm15, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 5);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 6);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 7);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm13 = ymm15;
ymm3 =  _mm256_set_epi32 (xmask[23],xmask[22],xmask[21],xmask[20],xmask[19],xmask[18],xmask[17],xmask[16]); 
ymm15 =  _mm256_maskload_ps (xptr + 16, ymm3);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 8);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 9);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 10);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 11);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm6 = _mm256_broadcast_ss(cptr + 12);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 13);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_maskstore_ps (yptr, ymm12, ymm8);}
}

void THFloatVector_conv1dk15(float * y, float * x, float * c, float alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 14;
float *yptr, *xptr, *cptr;
__m256 ymm0, ymm1, ymm2, ymm6, ymm8, ymm9, ymm10, ymm11, ymm13, ymm14, ymm15;
float ctmp[15];
for(j = 0; j < 15; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-8) && i <= (xn - 24); i+=8){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_ps(yptr);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
ymm13 = _mm256_loadu_ps(xptr);
ymm15 = _mm256_loadu_ps(xptr + 8);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm2 = _mm256_shuffle_ps(ymm1,  ymm15, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 5);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 6);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 7);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm13 = ymm15;
ymm15 = _mm256_loadu_ps(xptr + 16);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 8);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 9);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 10);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 11);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm6 = _mm256_broadcast_ss(cptr + 12);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 13);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 14);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_storeu_ps(yptr, ymm8);}
for(; i < n; i+=8){
int xmask[24] = {0};
int ymask[8] = {0};
int m;
for(m = 0; (m + i < n) && (m < 8); m++){
ymask[m] = 0xFFFFFFFF;
xmask[m] = 0xFFFFFFFF;}
for(; (m + i) < xn && (m < 24); m++){
xmask[m] = 0xFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi32 (ymask[7], ymask[6], ymask[5], ymask[4], ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_ps (yptr, ymm12);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
__m256i ymm3 =  _mm256_set_epi32 (xmask[7], xmask[6], xmask[5], xmask[4], xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm13 =  _mm256_maskload_ps (xptr, ymm3);
ymm3 =  _mm256_set_epi32 (xmask[15],xmask[14],xmask[13],xmask[12],xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm15 =  _mm256_maskload_ps (xptr + 8, ymm3);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm2 = _mm256_shuffle_ps(ymm1,  ymm15, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 5);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 6);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 7);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm13 = ymm15;
ymm3 =  _mm256_set_epi32 (xmask[23],xmask[22],xmask[21],xmask[20],xmask[19],xmask[18],xmask[17],xmask[16]); 
ymm15 =  _mm256_maskload_ps (xptr + 16, ymm3);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 8);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 9);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 10);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 11);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm6 = _mm256_broadcast_ss(cptr + 12);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 13);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 14);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_maskstore_ps (yptr, ymm12, ymm8);}
}

void THFloatVector_conv1dk16(float * y, float * x, float * c, float alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 15;
float *yptr, *xptr, *cptr;
__m256 ymm0, ymm1, ymm2, ymm6, ymm8, ymm9, ymm10, ymm11, ymm13, ymm14, ymm15;
float ctmp[16];
for(j = 0; j < 16; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-8) && i <= (xn - 24); i+=8){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_ps(yptr);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
ymm13 = _mm256_loadu_ps(xptr);
ymm15 = _mm256_loadu_ps(xptr + 8);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm2 = _mm256_shuffle_ps(ymm1,  ymm15, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 5);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 6);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 7);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm13 = ymm15;
ymm15 = _mm256_loadu_ps(xptr + 16);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 8);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 9);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 10);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 11);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm2 = _mm256_shuffle_ps(ymm1,  ymm15, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 12);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 13);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 14);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 15);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_storeu_ps(yptr, ymm8);}
for(; i < n; i+=8){
int xmask[24] = {0};
int ymask[8] = {0};
int m;
for(m = 0; (m + i < n) && (m < 8); m++){
ymask[m] = 0xFFFFFFFF;
xmask[m] = 0xFFFFFFFF;}
for(; (m + i) < xn && (m < 24); m++){
xmask[m] = 0xFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi32 (ymask[7], ymask[6], ymask[5], ymask[4], ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_ps (yptr, ymm12);
ymm9 = _mm256_setzero_ps();
ymm10 = _mm256_setzero_ps();
ymm11 = _mm256_setzero_ps();
__m256i ymm3 =  _mm256_set_epi32 (xmask[7], xmask[6], xmask[5], xmask[4], xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm13 =  _mm256_maskload_ps (xptr, ymm3);
ymm3 =  _mm256_set_epi32 (xmask[15],xmask[14],xmask[13],xmask[12],xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm15 =  _mm256_maskload_ps (xptr + 8, ymm3);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 0);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 1);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 2);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 3);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm2 = _mm256_shuffle_ps(ymm1,  ymm15, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 4);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 5);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 6);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 7);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm13 = ymm15;
ymm3 =  _mm256_set_epi32 (xmask[23],xmask[22],xmask[21],xmask[20],xmask[19],xmask[18],xmask[17],xmask[16]); 
ymm15 =  _mm256_maskload_ps (xptr + 16, ymm3);
ymm14 = _mm256_permute2f128_ps(ymm13, ymm15, 0x21);
ymm1 = _mm256_shuffle_ps(ymm13, ymm14, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm13, ymm1, 0x99);
ymm2 = _mm256_shuffle_ps(ymm1, ymm14, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 8);
ymm6 = _mm256_mul_ps(ymm6, ymm13);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 9);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 10);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 11);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm1 = _mm256_shuffle_ps(ymm14, ymm15, 0x4E);
ymm0 = _mm256_shuffle_ps(ymm14, ymm1,  0x99);
ymm2 = _mm256_shuffle_ps(ymm1,  ymm15, 0x99);
ymm6 = _mm256_broadcast_ss(cptr + 12);
ymm6 = _mm256_mul_ps(ymm6, ymm14);
ymm8 = _mm256_add_ps(ymm8, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 13);
ymm6 = _mm256_mul_ps(ymm6, ymm0);
ymm9 = _mm256_add_ps(ymm9, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 14);
ymm6 = _mm256_mul_ps(ymm6, ymm1);
ymm10 = _mm256_add_ps(ymm10, ymm6);
ymm6 = _mm256_broadcast_ss(cptr + 15);
ymm6 = _mm256_mul_ps(ymm6, ymm2);
ymm11 = _mm256_add_ps(ymm11, ymm6);
ymm8 = _mm256_add_ps(ymm8, ymm9);
ymm10 = _mm256_add_ps(ymm10, ymm11);
ymm8 = _mm256_add_ps(ymm8, ymm10);
_mm256_maskstore_ps (yptr, ymm12, ymm8);}
}

void THFloatVector_conv1d(float * y, float * x, float * c, float alpha, int n, int cn, int reverse){ 
int i; 
float * xptr = x;
float * cptr = c;
float alphatmp = alpha;
float coef;
int cnremained = cn;
int cnslice;
while(cnremained > 0){
if(cnremained > 16){
cnslice = 16;}
else { cnslice = cnremained;}
switch(cnslice){
case 1: 
case 2: 
if(reverse==0) { 
for(i = 0; i < cnslice; i++) {
coef = cptr[i] * alphatmp; 
THFloatVector_add_unrolled(y, xptr, coef, n); }} 
else { 
for(i = 0; i < cnslice; i++) {
coef = cptr[-i] * alphatmp; 
THFloatVector_add_unrolled(y, xptr, coef, n); }} 
break; 
case 3: THFloatVector_conv1dk3(y, xptr, cptr, alpha, n, reverse); break;
case 4: THFloatVector_conv1dk4(y, xptr, cptr, alpha, n, reverse); break;
case 5: THFloatVector_conv1dk5(y, xptr, cptr, alpha, n, reverse); break;
case 6: THFloatVector_conv1dk6(y, xptr, cptr, alpha, n, reverse); break;
case 7: THFloatVector_conv1dk7(y, xptr, cptr, alpha, n, reverse); break;
case 8: THFloatVector_conv1dk8(y, xptr, cptr, alpha, n, reverse); break;
case 9: THFloatVector_conv1dk9(y, xptr, cptr, alpha, n, reverse); break;
case 10: THFloatVector_conv1dk10(y, xptr, cptr, alpha, n, reverse); break;
case 11: THFloatVector_conv1dk11(y, xptr, cptr, alpha, n, reverse); break;
case 12: THFloatVector_conv1dk12(y, xptr, cptr, alpha, n, reverse); break;
case 13: THFloatVector_conv1dk13(y, xptr, cptr, alpha, n, reverse); break;
case 14: THFloatVector_conv1dk14(y, xptr, cptr, alpha, n, reverse); break;
case 15: THFloatVector_conv1dk15(y, xptr, cptr, alpha, n, reverse); break;
case 16: THFloatVector_conv1dk16(y, xptr, cptr, alpha, n, reverse); break;
default : 
if(reverse==0) { 
for(i = 0; i < cnslice; i++) {
coef = cptr[i] * alphatmp; 
THFloatVector_add_unrolled(y, xptr, coef, n); }} 
else { 
for(i = 0; i < cnslice; i++) {
coef = cptr[-i] * alphatmp; 
THFloatVector_add_unrolled(y, xptr, coef, n); }} 
break; 
}
cnremained -= cnslice;
xptr += cnslice;
if(reverse==0) { 
cptr += cnslice;}
else { 
cptr -= cnslice;}
}
}

void THDoubleVector_conv1dk3(double * y, double * x, double * c, double alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 2;
double *yptr, *xptr, *cptr;
__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm6, ymm8, ymm9, ymm10, ymm11;
double ctmp[3];
for(j = 0; j < 3; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-4) && i <= (xn - 8); i+=4){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_pd(yptr);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
ymm0 = _mm256_loadu_pd(xptr);
ymm4 = _mm256_loadu_pd(xptr + 4);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_storeu_pd(yptr, ymm8);}
for(; i < n; i+=4){
long long xmask[8] = {0};
long long ymask[4] = {0};
int m;
for(m = 0; (m + i < n) && (m < 4); m++){
ymask[m] = 0xFFFFFFFFFFFFFFFF;
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
for(; (m + i) < xn && (m < 8); m++){
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi64x (ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_pd (yptr, ymm12);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
__m256i ymm13 =  _mm256_set_epi64x (xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm0 =  _mm256_maskload_pd (xptr, ymm13);
ymm13 =  _mm256_set_epi64x (xmask[7],xmask[6],xmask[5],xmask[4]); 
ymm4 =  _mm256_maskload_pd (xptr + 4, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_maskstore_pd (yptr, ymm12, ymm8);}
}

void THDoubleVector_conv1dk4(double * y, double * x, double * c, double alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 3;
double *yptr, *xptr, *cptr;
__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm6, ymm8, ymm9, ymm10, ymm11;
double ctmp[4];
for(j = 0; j < 4; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-4) && i <= (xn - 8); i+=4){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_pd(yptr);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
ymm0 = _mm256_loadu_pd(xptr);
ymm4 = _mm256_loadu_pd(xptr + 4);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_storeu_pd(yptr, ymm8);}
for(; i < n; i+=4){
long long xmask[8] = {0};
long long ymask[4] = {0};
int m;
for(m = 0; (m + i < n) && (m < 4); m++){
ymask[m] = 0xFFFFFFFFFFFFFFFF;
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
for(; (m + i) < xn && (m < 8); m++){
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi64x (ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_pd (yptr, ymm12);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
__m256i ymm13 =  _mm256_set_epi64x (xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm0 =  _mm256_maskload_pd (xptr, ymm13);
ymm13 =  _mm256_set_epi64x (xmask[7],xmask[6],xmask[5],xmask[4]); 
ymm4 =  _mm256_maskload_pd (xptr + 4, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_maskstore_pd (yptr, ymm12, ymm8);}
}

void THDoubleVector_conv1dk5(double * y, double * x, double * c, double alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 4;
double *yptr, *xptr, *cptr;
__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm6, ymm8, ymm9, ymm10, ymm11;
double ctmp[5];
for(j = 0; j < 5; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-4) && i <= (xn - 8); i+=4){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_pd(yptr);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
ymm0 = _mm256_loadu_pd(xptr);
ymm4 = _mm256_loadu_pd(xptr + 4);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_storeu_pd(yptr, ymm8);}
for(; i < n; i+=4){
long long xmask[8] = {0};
long long ymask[4] = {0};
int m;
for(m = 0; (m + i < n) && (m < 4); m++){
ymask[m] = 0xFFFFFFFFFFFFFFFF;
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
for(; (m + i) < xn && (m < 8); m++){
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi64x (ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_pd (yptr, ymm12);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
__m256i ymm13 =  _mm256_set_epi64x (xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm0 =  _mm256_maskload_pd (xptr, ymm13);
ymm13 =  _mm256_set_epi64x (xmask[7],xmask[6],xmask[5],xmask[4]); 
ymm4 =  _mm256_maskload_pd (xptr + 4, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_maskstore_pd (yptr, ymm12, ymm8);}
}

void THDoubleVector_conv1dk6(double * y, double * x, double * c, double alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 5;
double *yptr, *xptr, *cptr;
__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm6, ymm8, ymm9, ymm10, ymm11;
double ctmp[6];
for(j = 0; j < 6; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-4) && i <= (xn - 12); i+=4){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_pd(yptr);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
ymm0 = _mm256_loadu_pd(xptr);
ymm4 = _mm256_loadu_pd(xptr + 4);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm4 = _mm256_loadu_pd(xptr + 8);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 5);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_storeu_pd(yptr, ymm8);}
for(; i < n; i+=4){
long long xmask[12] = {0};
long long ymask[4] = {0};
int m;
for(m = 0; (m + i < n) && (m < 4); m++){
ymask[m] = 0xFFFFFFFFFFFFFFFF;
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
for(; (m + i) < xn && (m < 12); m++){
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi64x (ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_pd (yptr, ymm12);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
__m256i ymm13 =  _mm256_set_epi64x (xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm0 =  _mm256_maskload_pd (xptr, ymm13);
ymm13 =  _mm256_set_epi64x (xmask[7],xmask[6],xmask[5],xmask[4]); 
ymm4 =  _mm256_maskload_pd (xptr + 4, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm13 =  _mm256_set_epi64x (xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm4 =  _mm256_maskload_pd (xptr + 8, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 5);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_maskstore_pd (yptr, ymm12, ymm8);}
}

void THDoubleVector_conv1dk7(double * y, double * x, double * c, double alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 6;
double *yptr, *xptr, *cptr;
__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm6, ymm8, ymm9, ymm10, ymm11;
double ctmp[7];
for(j = 0; j < 7; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-4) && i <= (xn - 12); i+=4){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_pd(yptr);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
ymm0 = _mm256_loadu_pd(xptr);
ymm4 = _mm256_loadu_pd(xptr + 4);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm4 = _mm256_loadu_pd(xptr + 8);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 5);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 6);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_storeu_pd(yptr, ymm8);}
for(; i < n; i+=4){
long long xmask[12] = {0};
long long ymask[4] = {0};
int m;
for(m = 0; (m + i < n) && (m < 4); m++){
ymask[m] = 0xFFFFFFFFFFFFFFFF;
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
for(; (m + i) < xn && (m < 12); m++){
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi64x (ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_pd (yptr, ymm12);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
__m256i ymm13 =  _mm256_set_epi64x (xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm0 =  _mm256_maskload_pd (xptr, ymm13);
ymm13 =  _mm256_set_epi64x (xmask[7],xmask[6],xmask[5],xmask[4]); 
ymm4 =  _mm256_maskload_pd (xptr + 4, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm13 =  _mm256_set_epi64x (xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm4 =  _mm256_maskload_pd (xptr + 8, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 5);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 6);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_maskstore_pd (yptr, ymm12, ymm8);}
}

void THDoubleVector_conv1dk8(double * y, double * x, double * c, double alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 7;
double *yptr, *xptr, *cptr;
__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm6, ymm8, ymm9, ymm10, ymm11;
double ctmp[8];
for(j = 0; j < 8; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-4) && i <= (xn - 12); i+=4){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_pd(yptr);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
ymm0 = _mm256_loadu_pd(xptr);
ymm4 = _mm256_loadu_pd(xptr + 4);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm4 = _mm256_loadu_pd(xptr + 8);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 5);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 6);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 7);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_storeu_pd(yptr, ymm8);}
for(; i < n; i+=4){
long long xmask[12] = {0};
long long ymask[4] = {0};
int m;
for(m = 0; (m + i < n) && (m < 4); m++){
ymask[m] = 0xFFFFFFFFFFFFFFFF;
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
for(; (m + i) < xn && (m < 12); m++){
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi64x (ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_pd (yptr, ymm12);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
__m256i ymm13 =  _mm256_set_epi64x (xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm0 =  _mm256_maskload_pd (xptr, ymm13);
ymm13 =  _mm256_set_epi64x (xmask[7],xmask[6],xmask[5],xmask[4]); 
ymm4 =  _mm256_maskload_pd (xptr + 4, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm13 =  _mm256_set_epi64x (xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm4 =  _mm256_maskload_pd (xptr + 8, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 5);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 6);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 7);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_maskstore_pd (yptr, ymm12, ymm8);}
}

void THDoubleVector_conv1dk9(double * y, double * x, double * c, double alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 8;
double *yptr, *xptr, *cptr;
__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm6, ymm8, ymm9, ymm10, ymm11;
double ctmp[9];
for(j = 0; j < 9; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-4) && i <= (xn - 12); i+=4){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_pd(yptr);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
ymm0 = _mm256_loadu_pd(xptr);
ymm4 = _mm256_loadu_pd(xptr + 4);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm4 = _mm256_loadu_pd(xptr + 8);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 5);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 6);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 7);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm6 = _mm256_broadcast_sd(cptr + 8);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_storeu_pd(yptr, ymm8);}
for(; i < n; i+=4){
long long xmask[12] = {0};
long long ymask[4] = {0};
int m;
for(m = 0; (m + i < n) && (m < 4); m++){
ymask[m] = 0xFFFFFFFFFFFFFFFF;
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
for(; (m + i) < xn && (m < 12); m++){
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi64x (ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_pd (yptr, ymm12);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
__m256i ymm13 =  _mm256_set_epi64x (xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm0 =  _mm256_maskload_pd (xptr, ymm13);
ymm13 =  _mm256_set_epi64x (xmask[7],xmask[6],xmask[5],xmask[4]); 
ymm4 =  _mm256_maskload_pd (xptr + 4, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm13 =  _mm256_set_epi64x (xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm4 =  _mm256_maskload_pd (xptr + 8, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 5);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 6);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 7);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm6 = _mm256_broadcast_sd(cptr + 8);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_maskstore_pd (yptr, ymm12, ymm8);}
}

void THDoubleVector_conv1dk10(double * y, double * x, double * c, double alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 9;
double *yptr, *xptr, *cptr;
__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm6, ymm8, ymm9, ymm10, ymm11;
double ctmp[10];
for(j = 0; j < 10; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-4) && i <= (xn - 16); i+=4){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_pd(yptr);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
ymm0 = _mm256_loadu_pd(xptr);
ymm4 = _mm256_loadu_pd(xptr + 4);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm4 = _mm256_loadu_pd(xptr + 8);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 5);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 6);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 7);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm4 = _mm256_loadu_pd(xptr + 12);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 8);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 9);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_storeu_pd(yptr, ymm8);}
for(; i < n; i+=4){
long long xmask[16] = {0};
long long ymask[4] = {0};
int m;
for(m = 0; (m + i < n) && (m < 4); m++){
ymask[m] = 0xFFFFFFFFFFFFFFFF;
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
for(; (m + i) < xn && (m < 16); m++){
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi64x (ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_pd (yptr, ymm12);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
__m256i ymm13 =  _mm256_set_epi64x (xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm0 =  _mm256_maskload_pd (xptr, ymm13);
ymm13 =  _mm256_set_epi64x (xmask[7],xmask[6],xmask[5],xmask[4]); 
ymm4 =  _mm256_maskload_pd (xptr + 4, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm13 =  _mm256_set_epi64x (xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm4 =  _mm256_maskload_pd (xptr + 8, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 5);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 6);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 7);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm13 =  _mm256_set_epi64x (xmask[15],xmask[14],xmask[13],xmask[12]); 
ymm4 =  _mm256_maskload_pd (xptr + 12, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 8);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 9);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_maskstore_pd (yptr, ymm12, ymm8);}
}

void THDoubleVector_conv1dk11(double * y, double * x, double * c, double alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 10;
double *yptr, *xptr, *cptr;
__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm6, ymm8, ymm9, ymm10, ymm11;
double ctmp[11];
for(j = 0; j < 11; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-4) && i <= (xn - 16); i+=4){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_pd(yptr);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
ymm0 = _mm256_loadu_pd(xptr);
ymm4 = _mm256_loadu_pd(xptr + 4);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm4 = _mm256_loadu_pd(xptr + 8);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 5);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 6);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 7);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm4 = _mm256_loadu_pd(xptr + 12);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 8);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 9);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 10);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_storeu_pd(yptr, ymm8);}
for(; i < n; i+=4){
long long xmask[16] = {0};
long long ymask[4] = {0};
int m;
for(m = 0; (m + i < n) && (m < 4); m++){
ymask[m] = 0xFFFFFFFFFFFFFFFF;
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
for(; (m + i) < xn && (m < 16); m++){
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi64x (ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_pd (yptr, ymm12);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
__m256i ymm13 =  _mm256_set_epi64x (xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm0 =  _mm256_maskload_pd (xptr, ymm13);
ymm13 =  _mm256_set_epi64x (xmask[7],xmask[6],xmask[5],xmask[4]); 
ymm4 =  _mm256_maskload_pd (xptr + 4, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm13 =  _mm256_set_epi64x (xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm4 =  _mm256_maskload_pd (xptr + 8, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 5);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 6);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 7);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm13 =  _mm256_set_epi64x (xmask[15],xmask[14],xmask[13],xmask[12]); 
ymm4 =  _mm256_maskload_pd (xptr + 12, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 8);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 9);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 10);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_maskstore_pd (yptr, ymm12, ymm8);}
}

void THDoubleVector_conv1dk12(double * y, double * x, double * c, double alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 11;
double *yptr, *xptr, *cptr;
__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm6, ymm8, ymm9, ymm10, ymm11;
double ctmp[12];
for(j = 0; j < 12; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-4) && i <= (xn - 16); i+=4){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_pd(yptr);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
ymm0 = _mm256_loadu_pd(xptr);
ymm4 = _mm256_loadu_pd(xptr + 4);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm4 = _mm256_loadu_pd(xptr + 8);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 5);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 6);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 7);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm4 = _mm256_loadu_pd(xptr + 12);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 8);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 9);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 10);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 11);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_storeu_pd(yptr, ymm8);}
for(; i < n; i+=4){
long long xmask[16] = {0};
long long ymask[4] = {0};
int m;
for(m = 0; (m + i < n) && (m < 4); m++){
ymask[m] = 0xFFFFFFFFFFFFFFFF;
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
for(; (m + i) < xn && (m < 16); m++){
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi64x (ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_pd (yptr, ymm12);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
__m256i ymm13 =  _mm256_set_epi64x (xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm0 =  _mm256_maskload_pd (xptr, ymm13);
ymm13 =  _mm256_set_epi64x (xmask[7],xmask[6],xmask[5],xmask[4]); 
ymm4 =  _mm256_maskload_pd (xptr + 4, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm13 =  _mm256_set_epi64x (xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm4 =  _mm256_maskload_pd (xptr + 8, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 5);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 6);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 7);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm13 =  _mm256_set_epi64x (xmask[15],xmask[14],xmask[13],xmask[12]); 
ymm4 =  _mm256_maskload_pd (xptr + 12, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 8);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 9);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 10);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 11);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_maskstore_pd (yptr, ymm12, ymm8);}
}

void THDoubleVector_conv1dk13(double * y, double * x, double * c, double alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 12;
double *yptr, *xptr, *cptr;
__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm6, ymm8, ymm9, ymm10, ymm11;
double ctmp[13];
for(j = 0; j < 13; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-4) && i <= (xn - 16); i+=4){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_pd(yptr);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
ymm0 = _mm256_loadu_pd(xptr);
ymm4 = _mm256_loadu_pd(xptr + 4);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm4 = _mm256_loadu_pd(xptr + 8);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 5);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 6);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 7);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm4 = _mm256_loadu_pd(xptr + 12);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 8);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 9);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 10);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 11);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm6 = _mm256_broadcast_sd(cptr + 12);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_storeu_pd(yptr, ymm8);}
for(; i < n; i+=4){
long long xmask[16] = {0};
long long ymask[4] = {0};
int m;
for(m = 0; (m + i < n) && (m < 4); m++){
ymask[m] = 0xFFFFFFFFFFFFFFFF;
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
for(; (m + i) < xn && (m < 16); m++){
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi64x (ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_pd (yptr, ymm12);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
__m256i ymm13 =  _mm256_set_epi64x (xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm0 =  _mm256_maskload_pd (xptr, ymm13);
ymm13 =  _mm256_set_epi64x (xmask[7],xmask[6],xmask[5],xmask[4]); 
ymm4 =  _mm256_maskload_pd (xptr + 4, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm13 =  _mm256_set_epi64x (xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm4 =  _mm256_maskload_pd (xptr + 8, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 5);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 6);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 7);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm13 =  _mm256_set_epi64x (xmask[15],xmask[14],xmask[13],xmask[12]); 
ymm4 =  _mm256_maskload_pd (xptr + 12, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 8);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 9);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 10);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 11);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm6 = _mm256_broadcast_sd(cptr + 12);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_maskstore_pd (yptr, ymm12, ymm8);}
}

void THDoubleVector_conv1dk14(double * y, double * x, double * c, double alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 13;
double *yptr, *xptr, *cptr;
__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm6, ymm8, ymm9, ymm10, ymm11;
double ctmp[14];
for(j = 0; j < 14; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-4) && i <= (xn - 20); i+=4){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_pd(yptr);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
ymm0 = _mm256_loadu_pd(xptr);
ymm4 = _mm256_loadu_pd(xptr + 4);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm4 = _mm256_loadu_pd(xptr + 8);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 5);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 6);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 7);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm4 = _mm256_loadu_pd(xptr + 12);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 8);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 9);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 10);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 11);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm4 = _mm256_loadu_pd(xptr + 16);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 12);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 13);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_storeu_pd(yptr, ymm8);}
for(; i < n; i+=4){
long long xmask[20] = {0};
long long ymask[4] = {0};
int m;
for(m = 0; (m + i < n) && (m < 4); m++){
ymask[m] = 0xFFFFFFFFFFFFFFFF;
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
for(; (m + i) < xn && (m < 20); m++){
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi64x (ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_pd (yptr, ymm12);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
__m256i ymm13 =  _mm256_set_epi64x (xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm0 =  _mm256_maskload_pd (xptr, ymm13);
ymm13 =  _mm256_set_epi64x (xmask[7],xmask[6],xmask[5],xmask[4]); 
ymm4 =  _mm256_maskload_pd (xptr + 4, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm13 =  _mm256_set_epi64x (xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm4 =  _mm256_maskload_pd (xptr + 8, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 5);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 6);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 7);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm13 =  _mm256_set_epi64x (xmask[15],xmask[14],xmask[13],xmask[12]); 
ymm4 =  _mm256_maskload_pd (xptr + 12, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 8);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 9);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 10);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 11);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm13 =  _mm256_set_epi64x (xmask[19],xmask[18],xmask[17],xmask[16]); 
ymm4 =  _mm256_maskload_pd (xptr + 16, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 12);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 13);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_maskstore_pd (yptr, ymm12, ymm8);}
}

void THDoubleVector_conv1dk15(double * y, double * x, double * c, double alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 14;
double *yptr, *xptr, *cptr;
__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm6, ymm8, ymm9, ymm10, ymm11;
double ctmp[15];
for(j = 0; j < 15; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-4) && i <= (xn - 20); i+=4){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_pd(yptr);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
ymm0 = _mm256_loadu_pd(xptr);
ymm4 = _mm256_loadu_pd(xptr + 4);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm4 = _mm256_loadu_pd(xptr + 8);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 5);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 6);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 7);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm4 = _mm256_loadu_pd(xptr + 12);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 8);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 9);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 10);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 11);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm4 = _mm256_loadu_pd(xptr + 16);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 12);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 13);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 14);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_storeu_pd(yptr, ymm8);}
for(; i < n; i+=4){
long long xmask[20] = {0};
long long ymask[4] = {0};
int m;
for(m = 0; (m + i < n) && (m < 4); m++){
ymask[m] = 0xFFFFFFFFFFFFFFFF;
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
for(; (m + i) < xn && (m < 20); m++){
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi64x (ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_pd (yptr, ymm12);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
__m256i ymm13 =  _mm256_set_epi64x (xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm0 =  _mm256_maskload_pd (xptr, ymm13);
ymm13 =  _mm256_set_epi64x (xmask[7],xmask[6],xmask[5],xmask[4]); 
ymm4 =  _mm256_maskload_pd (xptr + 4, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm13 =  _mm256_set_epi64x (xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm4 =  _mm256_maskload_pd (xptr + 8, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 5);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 6);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 7);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm13 =  _mm256_set_epi64x (xmask[15],xmask[14],xmask[13],xmask[12]); 
ymm4 =  _mm256_maskload_pd (xptr + 12, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 8);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 9);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 10);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 11);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm13 =  _mm256_set_epi64x (xmask[19],xmask[18],xmask[17],xmask[16]); 
ymm4 =  _mm256_maskload_pd (xptr + 16, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 12);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 13);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 14);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_maskstore_pd (yptr, ymm12, ymm8);}
}

void THDoubleVector_conv1dk16(double * y, double * x, double * c, double alpha, unsigned int n, char reverse){
int i, j;
int xn = n + 15;
double *yptr, *xptr, *cptr;
__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm6, ymm8, ymm9, ymm10, ymm11;
double ctmp[16];
for(j = 0; j < 16; j++){
if(reverse)
ctmp[j] = c[-j] * alpha;
else
ctmp[j] = c[j] * alpha;}
cptr = &ctmp[0];
for(i = 0; i <= (n-4) && i <= (xn - 20); i+=4){
yptr = y + i;
xptr = x + i;
ymm8 = _mm256_loadu_pd(yptr);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
ymm0 = _mm256_loadu_pd(xptr);
ymm4 = _mm256_loadu_pd(xptr + 4);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm4 = _mm256_loadu_pd(xptr + 8);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 5);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 6);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 7);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm4 = _mm256_loadu_pd(xptr + 12);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 8);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 9);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 10);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 11);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm4 = _mm256_loadu_pd(xptr + 16);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 12);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 13);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 14);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 15);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_storeu_pd(yptr, ymm8);}
for(; i < n; i+=4){
long long xmask[20] = {0};
long long ymask[4] = {0};
int m;
for(m = 0; (m + i < n) && (m < 4); m++){
ymask[m] = 0xFFFFFFFFFFFFFFFF;
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
for(; (m + i) < xn && (m < 20); m++){
xmask[m] = 0xFFFFFFFFFFFFFFFF;}
yptr = y + i;
xptr = x + i;
__m256i ymm12 =  _mm256_set_epi64x (ymask[3], ymask[2], ymask[1], ymask[0]); 
ymm8 =  _mm256_maskload_pd (yptr, ymm12);
ymm9 = _mm256_setzero_pd();
ymm10 = _mm256_setzero_pd();
ymm11 = _mm256_setzero_pd();
__m256i ymm13 =  _mm256_set_epi64x (xmask[3], xmask[2], xmask[1], xmask[0]); 
ymm0 =  _mm256_maskload_pd (xptr, ymm13);
ymm13 =  _mm256_set_epi64x (xmask[7],xmask[6],xmask[5],xmask[4]); 
ymm4 =  _mm256_maskload_pd (xptr + 4, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 0);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 1);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 2);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 3);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm13 =  _mm256_set_epi64x (xmask[11],xmask[10],xmask[9],xmask[8]); 
ymm4 =  _mm256_maskload_pd (xptr + 8, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 4);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 5);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 6);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 7);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm13 =  _mm256_set_epi64x (xmask[15],xmask[14],xmask[13],xmask[12]); 
ymm4 =  _mm256_maskload_pd (xptr + 12, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 8);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 9);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 10);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 11);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm0 = ymm4;
ymm13 =  _mm256_set_epi64x (xmask[19],xmask[18],xmask[17],xmask[16]); 
ymm4 =  _mm256_maskload_pd (xptr + 16, ymm13);
ymm2 = _mm256_permute2f128_pd(ymm0, ymm4, 0x21);
ymm1 = _mm256_shuffle_pd(ymm0, ymm2, 0x5);
ymm3 = _mm256_shuffle_pd(ymm2, ymm4, 0x5);
ymm6 = _mm256_broadcast_sd(cptr + 12);
ymm6 = _mm256_mul_pd(ymm6, ymm0);
ymm8 = _mm256_add_pd(ymm8, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 13);
ymm6 = _mm256_mul_pd(ymm6, ymm1);
ymm9 = _mm256_add_pd(ymm9, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 14);
ymm6 = _mm256_mul_pd(ymm6, ymm2);
ymm10 = _mm256_add_pd(ymm10, ymm6);
ymm6 = _mm256_broadcast_sd(cptr + 15);
ymm6 = _mm256_mul_pd(ymm6, ymm3);
ymm11 = _mm256_add_pd(ymm11, ymm6);
ymm8 = _mm256_add_pd(ymm8, ymm9);
ymm10 = _mm256_add_pd(ymm10, ymm11);
ymm8 = _mm256_add_pd(ymm8, ymm10);
_mm256_maskstore_pd (yptr, ymm12, ymm8);}
}

#define THDoubleVector_conv1d(y, x, c, alpha, n, cn, reverse){ \
int i;\
double * xptr = x;\
double * cptr = c;\
double alphatmp = alpha;\
double coef;\
int cnremained = cn;\
int cnslice;\
while(cnremained > 0){\
if(cnremained > 16){\
cnslice = 16;}\
else { cnslice = cnremained;}\
switch(cnslice){\
case 1: \
case 2: \
if(reverse==0) { \
for(i = 0; i < cnslice; i++) {\
coef = cptr[i] * alphatmp; \
THDoubleVector_add_unrolled(y, xptr, coef, n); }} \
else { \
for(i = 0; i < cnslice; i++) {\
coef = cptr[-i] * alphatmp; \
THDoubleVector_add_unrolled(y, xptr, coef, n); }} \
break; \
case 3: THDoubleVector_conv1dk3(y, xptr, cptr, alpha, n, reverse); break;\
case 4: THDoubleVector_conv1dk4(y, xptr, cptr, alpha, n, reverse); break;\
case 5: THDoubleVector_conv1dk5(y, xptr, cptr, alpha, n, reverse); break;\
case 6: THDoubleVector_conv1dk6(y, xptr, cptr, alpha, n, reverse); break;\
case 7: THDoubleVector_conv1dk7(y, xptr, cptr, alpha, n, reverse); break;\
case 8: THDoubleVector_conv1dk8(y, xptr, cptr, alpha, n, reverse); break;\
case 9: THDoubleVector_conv1dk9(y, xptr, cptr, alpha, n, reverse); break;\
case 10: THDoubleVector_conv1dk10(y, xptr, cptr, alpha, n, reverse); break;\
case 11: THDoubleVector_conv1dk11(y, xptr, cptr, alpha, n, reverse); break;\
case 12: THDoubleVector_conv1dk12(y, xptr, cptr, alpha, n, reverse); break;\
case 13: THDoubleVector_conv1dk13(y, xptr, cptr, alpha, n, reverse); break;\
case 14: THDoubleVector_conv1dk14(y, xptr, cptr, alpha, n, reverse); break;\
case 15: THDoubleVector_conv1dk15(y, xptr, cptr, alpha, n, reverse); break;\
case 16: THDoubleVector_conv1dk16(y, xptr, cptr, alpha, n, reverse); break;\
default : \
if(reverse==0) { \
for(i = 0; i < cnslice; i++) {\
coef = cptr[i] * alphatmp; \
THDoubleVector_add_unrolled(y, xptr, coef, n); }} \
else { \
for(i = 0; i < cnslice; i++) {\
coef = cptr[-i] * alphatmp; \
THDoubleVector_add_unrolled(y, xptr, coef, n); }} \
break; \
}\
cnremained -= cnslice;\
xptr += cnslice;\
if(reverse==0) { \
cptr += cnslice;}\
else { \
cptr -= cnslice;}\
}\
}
