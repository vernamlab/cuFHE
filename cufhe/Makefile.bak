## Compilers
CC = g++
CU = /usr/local/cuda/bin/nvcc
## Flags
FLAGS = -std=c++11 -O3
CU_FLAGS = -Wno-deprecated-gpu-targets
INC = -I./
##
dir_guard = @mkdir -p $(@D)

##
DIR_BIN = bin
DIR_OBJ = build
DIR_SRC = lib
CU_SRC = $(wildcard $(DIR_SRC)/details/*.cu) \
				 $(wildcard $(DIR_SRC)/ntt_gpu/*.cu) \
				 $(wildcard $(DIR_SRC)/*.cu)
CC_SRC = $(wildcard $(DIR_SRC)/details/*.cc) \
				 $(wildcard $(DIR_SRC)/*.cc)
CU_OBJ = $(patsubst $(DIR_SRC)/%,$(DIR_OBJ)/%,$(CU_SRC:.cu=.o))
CU_DEP = $(CU_OBJ:.o=.d)
CC_OBJ = $(patsubst $(DIR_SRC)/%,$(DIR_OBJ)/%,$(CC_SRC:.cc=.o))
CC_DEP = $(CC_OBJ:.o=.d)

##
targets: $(DIR_BIN)/test_api_cpu $(DIR_BIN)/test_api_gpu
##
all: $(targets)
cpu: $(DIR_BIN)/test_api_cpu
gpu: $(DIR_BIN)/test_api_gpu
clean:
	rm -r $(DIR_BIN) $(DIR_OBJ)
##
$(DIR_BIN)/test_api_cpu: $(CC_OBJ) $(DIR_OBJ)/test/test_api_cpu.o
	$(dir_guard)
	$(CC) $(FLAGS) -o $@ $(CC_OBJ) $(DIR_OBJ)/test/test_api_cpu.o
$(DIR_OBJ)/test/test_api_cpu.o: test/test_api_cpu.cc
	$(dir_guard)
	$(CC) $(FLAGS) $(INC) -M -o $(@:%.o=%.d) $<
	$(CC) $(FLAGS) $(INC) -c -o $@ $<

$(DIR_BIN)/test_api_gpu: $(DIR_OBJ)/test/test_api_gpu.o $(CU_OBJ) $(DIR_OBJ)/cufhe.o
	$(dir_guard)
	$(CU) $(FLAGS) -o $@ $(CU_OBJ) $(DIR_OBJ)/cufhe.o $(DIR_OBJ)/test/test_api_gpu.o
$(DIR_OBJ)/test/test_api_gpu.o: test/test_api_gpu.cu
	$(dir_guard)
	$(CU) $(FLAGS) $(INC) -M -o $(@:%.o=%.d) $<
	$(CU) $(FLAGS) $(INC) -c -o $@ $<

$(CU_OBJ): $(CU_SRC)
	$(dir_guard)
	$(CU) $(FLAGS) $(INC) -M -o $(@:%.o=%.d) \
			$(patsubst $(DIR_OBJ)/%,$(DIR_SRC)/%,$(@:%.o=%.cu))
	$(CU) $(FLAGS) $(INC) -c -o $@ \
			$(patsubst $(DIR_OBJ)/%,$(DIR_SRC)/%,$(@:%.o=%.cu))
$(CC_OBJ): $(CC_SRC)
	$(dir_guard)
	$(CC) $(FLAGS) $(INC) -M -o $(@:%.o=%.d) \
			$(patsubst $(DIR_OBJ)/%,$(DIR_SRC)/%,$(@:%.o=%.cc))
	$(CC) $(FLAGS) $(INC) -c -o $@ \
			$(patsubst $(DIR_OBJ)/%,$(DIR_SRC)/%,$(@:%.o=%.cc))
