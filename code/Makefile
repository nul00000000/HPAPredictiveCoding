FILES = src/main.c src/neuron.c src/lodepng.c src/spikeneuron.c
INCLUDE = include
OUT = main

all: $(OUT)

$(OUT): $(FILES)
	gcc -o $(OUT) $(FILES) -Iinclude -lm

run: $(OUT)
	./$(OUT) > output/out.txt

clean:
	rm $(OUT) $(OUT_PL)
