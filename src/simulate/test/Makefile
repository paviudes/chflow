# See https://gist.github.com/xuhdev/1873316.
CC = gcc
CFLAGS = -fPIC -Wall -Wextra -std=c17
LDFLAGS = -shared
RM = rm
LIBS = -lm
TARGET = test.so

$(TARGET):test.o
	$(CC) $(CFLAGS) $(LDFLAGS) $(LIBS) -o $(TARGET) test.o

obj/test.o: test.c Makefile
	$(CC) $(CFLAGS) -c test.c -o test.o

clean:
	$(RM) test.o $(TARGET)