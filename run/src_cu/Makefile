NVCC = nvcc
CFLAGS = -O3

TARGET = collusion
SRC = collusion.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET) results.bin

run: $(TARGET)
	./$(TARGET)

run-full: $(TARGET)
	./$(TARGET) 1000 500000 6

.PHONY: all clean run run-full
