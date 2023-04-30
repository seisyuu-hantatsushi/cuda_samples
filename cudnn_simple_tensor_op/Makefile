TARGET = simple_tensor_op

DEBUG_TARGET = $(TARGET)_debug
RELEASE_TARGET = $(TARGET)

C_OBJS = main.o
C_OBJS += add_test.o
C_OBJS += hadamard_product.o
#C_OBJS += matmul.o

CXX_OBJS =

INCDIRS = -I/usr/local/cuda/include
LIBS = -lcudnn -L/usr/local/cuda/lib64 -lcudart
include  ../Makefile.common
