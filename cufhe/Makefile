## Compilers
CC = g++
CU = /usr/local/cuda/bin/nvcc
## Flags
FLAGS = -std=c++11 -O3 -w
CU_FLAGS = -Wno-deprecated-gpu-targets
INC = -I./
## Python header files
PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
## Boost header files and library
BOOST_INCLUDE = /usr/include
BOOST_LIB = /usr/lib
## Boost + Python
PY_INC = -I./include -I$(PYTHON_INCLUDE) -I$(BOOST_INCLUDE) 
PY_LIB = -lpython$(PYTHON_VERSION) -lboost_python
##
dir_guard = @mkdir -p $(@D)

##
DIR_BIN = bin
DIR_OBJ = build
DIR_SRC = lib
DIR_PY = python
CU_SRC = $(wildcard $(DIR_SRC)/details/*.cu) \
				 $(wildcard $(DIR_SRC)/ntt_gpu/*.cu) \
				 $(wildcard $(DIR_SRC)/*.cu)
CC_SRC = $(wildcard $(DIR_SRC)/details/*.cc) \
				 $(wildcard $(DIR_SRC)/*.cc)
PY_SRC = $(wildcard $(DIR_PY)/*.cpp)
CU_OBJ = $(patsubst $(DIR_SRC)/%,$(DIR_OBJ)/%,$(CU_SRC:.cu=.o))
CU_DEP = $(CU_OBJ:.o=.d)
CC_OBJ = $(patsubst $(DIR_SRC)/%,$(DIR_OBJ)/%,$(CC_SRC:.cc=.o))
CC_DEP = $(CC_OBJ:.o=.d)

##
targets: $(DIR_BIN)/libcufhe_cpu.so $(DIR_BIN)/libcufhe_gpu.so \
	$(DIR_BIN)/test_api_cpu $(DIR_BIN)/test_api_gpu

##
all: $(targets)
cpu: $(DIR_BIN)/libcufhe_cpu.so $(DIR_BIN)/test_api_cpu
gpu: $(DIR_BIN)/libcufhe_gpu.so $(DIR_BIN)/test_api_gpu
python: $(DIR_BIN)/libcufhe_cpu.so $(DIR_PY)/pyfhe.so
clean:
	rm -r $(DIR_BIN) $(DIR_OBJ)

##
$(DIR_BIN)/test_api_cpu: $(DIR_OBJ)/test/test_api_cpu.o
	$(dir_guard)
	$(CC) $(FLAGS) -o $@ $(DIR_OBJ)/test/test_api_cpu.o -L$(DIR_BIN) -lcufhe_cpu

$(DIR_OBJ)/test/test_api_cpu.o: test/test_api_cpu.cc
	$(dir_guard)
	$(CC) $(FLAGS) $(INC) -M -o $(@:%.o=%.d) $<
	$(CC) $(FLAGS) $(INC) -c -o $@ $<

$(DIR_BIN)/test_api_gpu: $(DIR_OBJ)/test/test_api_gpu.o $(DIR_OBJ)/cufhe_io.o
	$(dir_guard)
	$(CU) $(FLAGS) -o $@ $(DIR_OBJ)/test/test_api_gpu.o $(DIR_OBJ)/cufhe_io.o -L$(DIR_BIN) -lcufhe_gpu

$(DIR_OBJ)/test/test_api_gpu.o: test/test_api_gpu.cu
	$(dir_guard)
	$(CU) $(FLAGS) $(INC) -M -o $(@:%.o=%.d) $<
	$(CU) $(FLAGS) $(INC) -c -o $@ $<

$(DIR_BIN)/libcufhe_cpu.so: $(CC_OBJ)
	$(dir_guard)
	$(CU) $(FLAGS) -shared -o $@ $(CC_OBJ)

$(DIR_BIN)/libcufhe_gpu.so: $(CU_OBJ) $(DIR_OBJ)/cufhe.o
	$(dir_guard)
	$(CU) $(FLAGS) -shared -o $@ $(CU_OBJ) $(DIR_OBJ)/cufhe.o

$(CC_OBJ): $(CC_SRC)
	$(dir_guard)
	$(CC) $(FLAGS) $(INC) -M -o $(@:%.o=%.d) \
			$(patsubst $(DIR_OBJ)/%,$(DIR_SRC)/%,$(@:%.o=%.cc))
	$(CC) $(FLAGS) $(INC) -fPIC -c -o $@ \
			$(patsubst $(DIR_OBJ)/%,$(DIR_SRC)/%,$(@:%.o=%.cc))

$(CU_OBJ): $(CU_SRC)
	$(dir_guard)
	$(CU) $(FLAGS) $(INC) -M -o $(@:%.o=%.d) \
			$(patsubst $(DIR_OBJ)/%,$(DIR_SRC)/%,$(@:%.o=%.cu))
	$(CU) $(FLAGS) $(INC) -c -o $@ \
			$(patsubst $(DIR_OBJ)/%,$(DIR_SRC)/%,$(@:%.o=%.cu)) -Xcompiler '-fPIC'

$(DIR_PY)/pyfhe.so: $(DIR_PY)/pyfhe.o
	$(CC) -shared $(DIR_PY)/pyfhe.o $(PY_LIB) -L$(DIR_BIN) -lcufhe_cpu -o $@

$(DIR_PY)/pyfhe.o: $(PY_SRC)
	$(CC) $(INC) $(PY_INC) -fPIC -c $(PY_SRC) $(FLAGS) -o $@
