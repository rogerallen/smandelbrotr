# inspired by https://stackoverflow.com/questions/2481269/how-to-make-a-simple-c-makefile/2481326
# that led to http://make.mad-scientist.net/papers/advanced-auto-dependency-generation/#combine

# use 'make DEBUG=1' for debug build
DEBUG ?= 1
ifeq (DEBUG, 1)
    OPT_FLAGS =-g -DDEBUG
else
    OPT_FLAGS =-O3 -DNDEBUG
endif

CUDA_DIR=/usr/local/cuda
# adjust for your GPU
CUDA_GENCODE=-gencode arch=compute_60,code=sm_60
NVCC=$(CUDA_DIR)/bin/nvcc
NVCC_FLAGS=$(OPT_FLAGS) -I$(CUDA_DIR)/include $(CUDA_GENCODE)

CXX=g++
CXXFLAGS=$(OPT_FLAGS) -std=c++11 -Werror -Wall -Iglm -I$(CUDA_DIR)/include
LDFLAGS=-L $(CUDA_DIR)/lib64
LDLIBS=-lsfml-graphics -lsfml-window -lsfml-system -lGL -lGLEW -lcudart -lcuda

CUDA_SRCS=mandelbrot.cu
SRCS=main.cpp app.cpp appCUDA.cpp appGL.cpp

OBJS=$(subst .cpp,.o,$(SRCS)) $(subst .cu,.o,$(CUDA_SRCS))

DEPDIR := .depend
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td

POSTCOMPILE = @mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d && touch $@

all: smandelbrotr

smandelbrotr: $(OBJS)
	$(CXX) $(LDFLAGS) -o smandelbrotr $(OBJS) $(LDLIBS)

clean:
	$(RM) $(OBJS) smandelbrotr
	$(RM) -r $(DEPDIR)

%.o : %.cpp
%.o : %.cpp $(DEPDIR)/%.d
	$(CXX) $(DEPFLAGS) $(CXXFLAGS) -c $<
	$(POSTCOMPILE)

# FIXME - cuda dependency generation
%.o : %.cu
	$(NVCC) $(NVCC_FLAGS) -c $<

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

include $(wildcard $(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS))))
