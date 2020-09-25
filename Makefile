all:
	@clang -mavx2 -O2 -funroll-loops -m64 -march=native -pedantic -mtune=native -fomit-frame-pointer -fwrapv  $(wildcard src/*.c) $(wildcard src/*.S) -o test_bench
clean:
	@rm -f test_bench
