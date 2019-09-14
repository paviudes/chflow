# Use the compiler options and link commands in file:///opt/intel/documentation_2019/en/mkl/common/mkl_link_line_advisor.htm .
# The LAPACK variables can be defined in shell by running the following command.
#	source /opt/intel/compilers_and_libraries_2019/mac/bin/compilervars.sh intel64
# The Random number generator is defined in mt19937ar.c .
# See https://gist.github.com/xuhdev/1873316.
CC = gcc
OPTS = -O3 -ffast-math
CFLAGS = -fPIC -Wall -Wextra -std=c11 $(OPTS)
CFLAGS_MKL = -m64 -I${MKLROOT}/include
LIBS = -lm
LIBS_MKL = -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_rt -lpthread -lm -ldl
LDFLAGS = -shared
RM = rm
TARGET = bmark.so
SRC_DIR = qecc_concat

$(shell mkdir -p obj/)

$(TARGET):obj/main.o obj/sampling.o obj/constants.o obj/printfuns.o obj/mt19937ar.o obj/checks.o obj/logmetrics.o obj/memory.o obj/qecc.o obj/effective.o obj/benchmark.o obj/hybrid.o obj/linalg.o
	$(CC) $(CFLAGS) $(LDFLAGS) $(LIBS) -o $(TARGET) obj/main.o obj/sampling.o obj/constants.o obj/printfuns.o obj/mt19937ar.o obj/logmetrics.o obj/checks.o obj/memory.o obj/qecc.o obj/effective.o obj/benchmark.o obj/hybrid.o $(LIBS_MKL) obj/linalg.o

obj/main.o: $(SRC_DIR)/main.c Makefile
	$(CC) $(CFLAGS) -c $(SRC_DIR)/main.c -o obj/main.o

obj/mt19937ar.o: $(SRC_DIR)/mt19937/mt19937ar.c $(SRC_DIR)/mt19937/mt19937ar.h Makefile
	$(CC) $(CFLAGS) -c $(SRC_DIR)/mt19937/mt19937ar.c -o obj/mt19937ar.o

obj/linalg.o: $(SRC_DIR)/linalg.c $(SRC_DIR)/linalg.h Makefile
	$(CC) $(CFLAGS) $(CFLAGS_MKL) -c $(SRC_DIR)/linalg.c -o obj/linalg.o

obj/benchmark.o: $(SRC_DIR)/benchmark.c $(SRC_DIR)/benchmark.h Makefile
	$(CC) $(CFLAGS) -c $(SRC_DIR)/benchmark.c -o obj/benchmark.o

obj/hybrid.o: $(SRC_DIR)/hybrid.c $(SRC_DIR)/hybrid.h Makefile
	$(CC) $(CFLAGS) -c $(SRC_DIR)/hybrid.c -o obj/hybrid.o

obj/effective.o: $(SRC_DIR)/effective.c $(SRC_DIR)/effective.h Makefile
	$(CC) $(CFLAGS) -c $(SRC_DIR)/effective.c -o obj/effective.o

obj/memory.o: $(SRC_DIR)/memory.c $(SRC_DIR)/memory.h Makefile
	$(CC) $(CFLAGS) -c $(SRC_DIR)/memory.c -o obj/memory.o

obj/qecc.o: $(SRC_DIR)/qecc.c $(SRC_DIR)/qecc.h Makefile
	$(CC) $(CFLAGS) -c $(SRC_DIR)/qecc.c -o obj/qecc.o

obj/logmetrics.o: $(SRC_DIR)/logmetrics.c $(SRC_DIR)/logmetrics.h Makefile
	$(CC) $(CFLAGS) -c $(SRC_DIR)/logmetrics.c -o obj/logmetrics.o

obj/checks.o: $(SRC_DIR)/checks.c $(SRC_DIR)/checks.h Makefile
	$(CC) $(CFLAGS) -c $(SRC_DIR)/checks.c -o obj/checks.o

obj/printfuns.o: $(SRC_DIR)/printfuns.c $(SRC_DIR)/printfuns.h Makefile
	$(CC) $(CFLAGS) -c $(SRC_DIR)/printfuns.c -o obj/printfuns.o

obj/constants.o: $(SRC_DIR)/constants.c $(SRC_DIR)/constants.h Makefile
	$(CC) $(CFLAGS) -c $(SRC_DIR)/constants.c -o obj/constants.o

obj/sampling.o: $(SRC_DIR)/sampling.c $(SRC_DIR)/sampling.h Makefile
	$(CC) $(CFLAGS) -c $(SRC_DIR)/sampling.c -o obj/sampling.o

clean:
	$(RM) $(TARGET) obj/main.o obj/sampling.o obj/constants.o obj/printfuns.o obj/mt19937ar.o obj/checks.o obj/logmetrics.o obj/memory.o obj/qecc.o obj/effective.o obj/benchmark.o obj/hybrid.o obj/linalg.o
