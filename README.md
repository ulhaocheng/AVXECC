# AVXECC

High-throughput elliptic curve cryptography software using Advanced Vector Extensions.

Current implementations: 
- X25519 using AVX2

### Copyright
Copyright © 2020 by University of Luxembourg.

### Software Authors
Hao Cheng, Johann Groszschaedl and Jiaqi Tian (University of Luxembourg).

### Compiler
Clang

You can modify the Makefile to use other compilers (e.g. GCC), but the performance of the software might be affected. 

Because we "tuned" the code with Clang.

### Test and Benchmark
```bash
    $ make
    $ ./test_bench
```

### Clean
```bash
    $ make clean
```

### LICENSE
GPLv3 (see details in LICENSE file)