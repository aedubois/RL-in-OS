CC = g++
CFLAGS = -Wall -O2
TARGET = code

all: $(TARGET)

$(TARGET): code.o
    $(CC) $(CFLAGS) -o $(TARGET) code.o

code.o: code.cpp
    $(CC) $(CFLAGS) -c code.cpp

clean:
    rm -f $(TARGET) code.o