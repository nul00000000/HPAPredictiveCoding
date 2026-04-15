FILES = src/main.c src/neuron.c src/lodepng.c
INCLUDE = include
OUT = main

include extra.make

all: $(OUT)

$(OUT): $(FILES)
	nvcc -ccbin $(CCBIN) -o $(OUT) $(FILES) -Iinclude -lm

run: $(OUT)
	./$(OUT) > output/out.txt

clean:
	rm $(OUT) $(OUT_PL)
