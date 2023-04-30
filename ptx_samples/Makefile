TARGET = ptx_sample

DEBUG_TARGET = $(TARGET)_debug
RELEASE_TARGET = $(TARGET)

C_OBJS = main.o

CXX_OBJS =

PTXO_OBJS = ptxjit_kernel.ptxo

INCDIRS = -I/usr/local/cuda/include
LIBS = -L/usr/local/cuda/lib64 -lcudart -lcuda

include  ../Makefile.common

