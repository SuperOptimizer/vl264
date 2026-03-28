// vl264-fuzz-decode.c — AFL++ fuzz target for decoder
// Feeds arbitrary bitstream data to the decoder to find crashes.
// Build: cmake -B build-fuzz -DVL264_BUILD_FUZZ=ON -DCMAKE_C_COMPILER=afl-clang-fast
// Run:   afl-fuzz -i corpus/ -o findings/ -- ./vl264_fuzz_decode @@
#include "vl264.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 2) return 1;

    FILE* f = fopen(argv[1], "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    if (size <= 0 || size > 16 * 1024 * 1024) { fclose(f); return 1; }
    fseek(f, 0, SEEK_SET);

    uint8_t* data = malloc((size_t)size);
    if (!data) { fclose(f); return 1; }
    fread(data, 1, (size_t)size, f);
    fclose(f);

    uint8_t* output = malloc(VL264_CHUNK_VOXELS);
    if (!output) { free(data); return 1; }

    vl264_dec* dec = vl264_dec_create();
    if (dec) {
        vl264_decode(dec, data, (size_t)size, NULL, NULL, output);
        vl264_dec_destroy(dec);
    }

    free(output);
    free(data);
    return 0;
}
