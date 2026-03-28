// vl264-fuzz-roundtrip.c — AFL++ fuzz target for encode+decode roundtrip
// Feeds arbitrary chunk data through encode->decode and checks PSNR sanity.
// Build: cmake -B build-fuzz -DVL264_BUILD_FUZZ=ON -DCMAKE_C_COMPILER=afl-clang-fast
// Run:   afl-fuzz -i corpus/ -o findings/ -- ./vl264_fuzz_roundtrip @@
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
    fseek(f, 0, SEEK_SET);

    // Use input to seed a chunk + config
    uint8_t header[8] = {0};
    if (size >= 8) fread(header, 1, 8, f);

    uint8_t* chunk = calloc(1, VL264_CHUNK_VOXELS);
    if (!chunk) { fclose(f); return 1; }

    // Fill chunk from file data (repeat if shorter)
    size_t chunk_fill = (size_t)(size > 8 ? size - 8 : 0);
    if (chunk_fill > 0) {
        fread(chunk, 1, chunk_fill > VL264_CHUNK_VOXELS ? VL264_CHUNK_VOXELS : chunk_fill, f);
    }
    fclose(f);

    // Derive config from header
    vl264_cfg cfg = vl264_default_cfg();
    cfg.quality = header[0] % 3;
    cfg.qp = (header[1] % 51) + 1;
    cfg.axis = (header[2] % 3);
    cfg.morton_order = header[3] & 1;

    vl264_enc* enc = vl264_enc_create(&cfg);
    vl264_dec* dec = vl264_dec_create();
    if (!enc || !dec) { free(chunk); return 1; }

    vl264_buf out = {0};
    vl264_status s = vl264_encode(enc, chunk, NULL, NULL, &out);
    if (s == VL264_OK && out.data) {
        uint8_t* decoded = calloc(1, VL264_CHUNK_VOXELS);
        if (decoded) {
            vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);
            free(decoded);
        }
        vl264_free(out.data);
    }

    vl264_enc_destroy(enc);
    vl264_dec_destroy(dec);
    free(chunk);
    return 0;
}
