
#include <stdio.h>
#include <stdlib.h>
#include <cudnn.h>
#include <cudnn_ops_infer.h>

extern int add_test(void);
extern int hadamard_product(void);
extern int matmul(void);

uint32_t xorshitf32(uint32_t *pState){
    uint32_t state = *pState;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    *pState = state;
    return *pState;
}

int main(int argc, char **argv) {

    add_test();
    hadamard_product();
    //matmul();

    return 0;
}
