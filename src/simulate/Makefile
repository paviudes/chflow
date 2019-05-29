# Use the compiler options and link commands in file:///opt/intel/documentation_2019/en/mkl/common/mkl_link_line_advisor.htm .
# The LAPACK variables can be defined in shell by running the following command.
#	source /opt/intel/compilers_and_libraries_2019/mac/bin/compilervars.sh intel64
# The Random number generator is defined in mt19937ar.c .
# See https://gist.github.com/xuhdev/1873316.
CC = icc
OPTS = -O3 -xavx
REPORT = -qopt-report-phase=vec -qopt-report=5
CFLAGS = -fPIC -Wall -Wextra -std=c11 $(OPTS)
# $(REPORT)
CFLAGS_MKL = -m64 -I${MKLROOT}/include
LIBS = -lm
LIBS_MKL = -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_rt -lpthread $(LIBS) -ldl
LDFLAGS = -shared
RM = rm
TARGET = bmark.so

$(shell mkdir -p obj/)

$(TARGET):obj/main.o obj/sampling.o obj/constants.o obj/printfuns.o obj/mt19937ar.o obj/checks.o obj/logmetrics.o obj/memory.o obj/qecc.o obj/effective.o obj/benchmark.o obj/linalg.o
	$(CC) $(CFLAGS) $(LDFLAGS) obj/main.o obj/sampling.o obj/constants.o obj/printfuns.o obj/mt19937ar.o obj/logmetrics.o obj/checks.o obj/memory.o obj/qecc.o obj/effective.o obj/benchmark.o obj/linalg.o $(LIBS) $(LIBS_MKL) -o $(TARGET)

obj/main.o: backend/main.c Makefile
	$(CC) -c $(CFLAGS) backend/main.c -o obj/main.o

obj/mt19937ar.o: backend/mt19937/mt19937ar.c backend/mt19937/mt19937ar.h Makefile
	$(CC) -c $(CFLAGS) backend/mt19937/mt19937ar.c -o obj/mt19937ar.o

obj/linalg.o: backend/linalg.c backend/linalg.h Makefile
	$(CC) -c $(CFLAGS) $(CFLAGS_MKL) backend/linalg.c -o obj/linalg.o

obj/benchmark.o: backend/benchmark.c backend/benchmark.h Makefile
	$(CC) -c $(CFLAGS) backend/benchmark.c -o obj/benchmark.o

obj/effective.o: backend/effective.c backend/effective.h Makefile
	$(CC) -c $(CFLAGS) backend/effective.c -o obj/effective.o

obj/memory.o: backend/memory.c backend/memory.h Makefile
	$(CC) -c $(CFLAGS) backend/memory.c -o obj/memory.o

obj/qecc.o: backend/qecc.c backend/qecc.h Makefile
	$(CC) -c $(CFLAGS) backend/qecc.c -o obj/qecc.o

obj/logmetrics.o: backend/logmetrics.c backend/logmetrics.h Makefile
	$(CC) -c $(CFLAGS) backend/logmetrics.c -o obj/logmetrics.o

obj/checks.o: backend/checks.c backend/checks.h Makefile
	$(CC) -c $(CFLAGS) backend/checks.c -o obj/checks.o

obj/printfuns.o: backend/printfuns.c backend/printfuns.h Makefile
	$(CC) -c $(CFLAGS) backend/printfuns.c -o obj/printfuns.o

obj/constants.o: backend/constants.c backend/constants.h Makefile
	$(CC) -c $(CFLAGS) backend/constants.c -o obj/constants.o

obj/sampling.o: backend/sampling.c backend/sampling.h Makefile
	$(CC) -c $(CFLAGS) backend/sampling.c -o obj/sampling.o

clean:
	$(RM) $(TARGET) obj/main.o obj/sampling.o obj/constants.o obj/printfuns.o obj/mt19937ar.o obj/checks.o obj/logmetrics.o obj/memory.o obj/qecc.o obj/effective.o obj/benchmark.o obj/linalg.o
