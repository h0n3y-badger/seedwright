# CPU build: plain C++ with OpenMP. GPU build: nvcc (sm_120). Both: no FMA contraction (bit-exactness).
CXXFLAGS = -O2 -std=c++17 -ffp-contract=off -fno-fast-math -Wall -Wno-unused-function
NVFLAGS  = -O3 -std=c++17 -arch=sm_120 -fmad=false --extended-lambda -Xcompiler -ffp-contract=off,-fopenmp
all: seedwright-cpu seedwright-gpu
seedwright-cpu: main.cu sw_core.h pack.h structures.h query.h query_parse.h jigsaw.h jigsaw_cli.h terrain.h crack_cli.h layout.h
	g++ $(CXXFLAGS) -fopenmp -x c++ -DSW_CPU_ONLY main.cu -o $@ -lcrypto
seedwright-gpu: main.cu sw_core.h pack.h structures.h query.h query_parse.h jigsaw.h jigsaw_cli.h terrain.h crack_cli.h layout.h
	nvcc $(NVFLAGS) main.cu -o $@ -lcrypto -lgomp
clean:
	rm -f seedwright-cpu seedwright-gpu
