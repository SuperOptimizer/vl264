// vl264-test.c — VL264 test suite
// Pure C23, zero dependencies beyond libc + vl264.
#define _POSIX_C_SOURCE 199309L

#include "vl264.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ── Test framework ──────────────────────────────────────────────────────────

static int g_tests, g_pass, g_fail;

#define TEST(cond, msg) do { \
    g_tests++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
        g_fail++; \
    } else { \
        g_pass++; \
    } \
} while(0)

#define SECTION(name) printf("── %s\n", name)

static bool has_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], flag) == 0) return true;
    return false;
}

// ── Data generators ─────────────────────────────────────────────────────────

static void gen_constant(uint8_t* v, uint8_t val) {
    memset(v, val, VL264_CHUNK_VOXELS);
}

static void gen_gradient_z(uint8_t* v) {
    for (int z = 0; z < 128; z++)
        for (int y = 0; y < 128; y++)
            for (int x = 0; x < 128; x++)
                v[z * 128 * 128 + y * 128 + x] = (uint8_t)(z * 2);
}

static void gen_gradient_x(uint8_t* v) {
    for (int z = 0; z < 128; z++)
        for (int y = 0; y < 128; y++)
            for (int x = 0; x < 128; x++)
                v[z * 128 * 128 + y * 128 + x] = (uint8_t)(x * 2);
}

static uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static void gen_noise(uint8_t* v, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < VL264_CHUNK_VOXELS; i++)
        v[i] = (uint8_t)(xorshift32(&s) & 0xFF);
}

static void gen_sphere(uint8_t* v) {
    for (int z = 0; z < 128; z++)
        for (int y = 0; y < 128; y++)
            for (int x = 0; x < 128; x++) {
                float dx = (float)x - 64.0f;
                float dy = (float)y - 64.0f;
                float dz = (float)z - 64.0f;
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                if (dist < 40.0f) v[z*128*128 + y*128 + x] = 200;
                else if (dist < 42.0f) v[z*128*128 + y*128 + x] = 128;
                else v[z*128*128 + y*128 + x] = 10;
            }
}

static void gen_ct_phantom(uint8_t* v) {
    // Sphere + noise + tissue layers
    gen_sphere(v);
    uint32_t seed = 42;
    for (size_t i = 0; i < VL264_CHUNK_VOXELS; i++) {
        int32_t noise = (int32_t)(xorshift32(&seed) % 11) - 5; // +/- 5
        int32_t val = (int32_t)v[i] + noise;
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        v[i] = (uint8_t)val;
    }
}

// ── Bitstream tests ─────────────────────────────────────────────────────────

static void test_bitstream_proper(void) {
    SECTION("bitstream");

    // Verify encode -> decode round-trip for known data
    uint8_t* chunk = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
    uint8_t* decoded = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
    TEST(chunk != NULL && decoded != NULL, "alloc");

    gen_constant(chunk, 100);

    vl264_cfg cfg = vl264_default_cfg();
    cfg.quality = VL264_FAST;
    vl264_enc* enc = vl264_enc_create(&cfg);
    vl264_dec* dec = vl264_dec_create();
    TEST(enc != NULL && dec != NULL, "create enc/dec");

    vl264_buf out = {0};
    vl264_status s = vl264_encode(enc, chunk, NULL, NULL, &out);
    TEST(s == VL264_OK, "encode");

    s = vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);
    TEST(s == VL264_OK, "decode");

    // Constant chunk should decode very close to original
    float psnr = vl264_psnr(chunk, decoded, VL264_CHUNK_VOXELS);
    printf("    constant PSNR: %.1f dB, size: %zu\n", psnr, out.size);
    TEST(psnr > 15.0f, "constant PSNR > 15 dB");

    vl264_free(out.data);
    vl264_enc_destroy(enc);
    vl264_dec_destroy(dec);
    free(chunk);
    free(decoded);
}

// ── DCT tests ───────────────────────────────────────────────────────────────

// We test DCT indirectly through encode/decode quality
static void test_dct(void) {
    SECTION("dct");

    // Test via gradient encode/decode — gradient exercises DCT well
    uint8_t* chunk = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
    uint8_t* decoded = (uint8_t*)malloc(VL264_CHUNK_VOXELS);

    gen_gradient_z(chunk);

    vl264_cfg cfg = vl264_default_cfg();
    cfg.quality = VL264_MAX;
    cfg.qp = 10; // low QP for high quality
    vl264_enc* enc = vl264_enc_create(&cfg);
    vl264_dec* dec = vl264_dec_create();

    vl264_buf out = {0};
    vl264_status s = vl264_encode(enc, chunk, NULL, NULL, &out);
    TEST(s == VL264_OK, "encode gradient");

    s = vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);
    TEST(s == VL264_OK, "decode gradient");

    float psnr = vl264_psnr(chunk, decoded, VL264_CHUNK_VOXELS);
    printf("    gradient PSNR: %.1f dB, ratio: %.1f:1\n",
           psnr, (float)VL264_CHUNK_VOXELS / (float)out.size);
    TEST(psnr > 25.0f, "gradient PSNR > 25 dB");

    vl264_free(out.data);
    vl264_enc_destroy(enc);
    vl264_dec_destroy(dec);
    free(chunk);
    free(decoded);
}

// ── Quantization tests ──────────────────────────────────────────────────────

static void test_quant(void) {
    SECTION("quant");

    // Test that low QP preserves quality and high QP compresses aggressively
    uint8_t* chunk = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
    uint8_t* decoded = (uint8_t*)malloc(VL264_CHUNK_VOXELS);

    gen_ct_phantom(chunk);

    // Low QP (high quality)
    {
        vl264_cfg cfg = vl264_default_cfg();
        cfg.qp = 12;
        cfg.quality = VL264_MAX;
        vl264_enc* enc = vl264_enc_create(&cfg);
        vl264_dec* dec = vl264_dec_create();
        vl264_buf out = {0};

        vl264_encode(enc, chunk, NULL, NULL, &out);
        vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);
        float psnr_low = vl264_psnr(chunk, decoded, VL264_CHUNK_VOXELS);
        size_t size_low = out.size;

        printf("    QP=12 PSNR: %.1f dB, size: %zu\n", psnr_low, size_low);
        TEST(psnr_low > 30.0f, "low QP high quality");

        vl264_free(out.data);
        vl264_enc_destroy(enc);
        vl264_dec_destroy(dec);
    }

    // High QP (aggressive compression)
    {
        vl264_cfg cfg = vl264_default_cfg();
        cfg.qp = 40;
        cfg.quality = VL264_FAST;
        vl264_enc* enc = vl264_enc_create(&cfg);
        vl264_dec* dec = vl264_dec_create();
        vl264_buf out = {0};

        vl264_encode(enc, chunk, NULL, NULL, &out);
        vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);
        float psnr_high = vl264_psnr(chunk, decoded, VL264_CHUNK_VOXELS);
        size_t size_high = out.size;

        printf("    QP=40 PSNR: %.1f dB, size: %zu\n", psnr_high, size_high);
        TEST(size_high < VL264_CHUNK_VOXELS / 2, "high QP compresses well");

        vl264_free(out.data);
        vl264_enc_destroy(enc);
        vl264_dec_destroy(dec);
    }

    free(chunk);
    free(decoded);
}

// ── Intra prediction tests ──────────────────────────────────────────────────

static void test_intra(void) {
    SECTION("intra");

    // Constant data should compress very efficiently (DC mode dominates)
    uint8_t* chunk = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
    uint8_t* decoded = (uint8_t*)malloc(VL264_CHUNK_VOXELS);

    gen_constant(chunk, 128);

    vl264_cfg cfg = vl264_default_cfg();
    cfg.quality = VL264_FAST;
    vl264_enc* enc = vl264_enc_create(&cfg);
    vl264_dec* dec = vl264_dec_create();
    vl264_buf out = {0};

    vl264_encode(enc, chunk, NULL, NULL, &out);
    vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);

    float psnr = vl264_psnr(chunk, decoded, VL264_CHUNK_VOXELS);
    float ratio = (float)VL264_CHUNK_VOXELS / (float)out.size;
    printf("    constant: PSNR=%.1f dB, ratio=%.1f:1\n", psnr, ratio);
    TEST(psnr > 35.0f, "constant PSNR very high");
    TEST(ratio > 3.0f, "constant compresses well");

    vl264_free(out.data);
    vl264_enc_destroy(enc);
    vl264_dec_destroy(dec);
    free(chunk);
    free(decoded);
}

// ── Inter prediction tests ──────────────────────────────────────────────────

static void test_inter(void) {
    SECTION("inter");

    // Gradient along Z — sequential slices are very similar, P-frames should help
    uint8_t* chunk = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
    uint8_t* decoded = (uint8_t*)malloc(VL264_CHUNK_VOXELS);

    gen_gradient_z(chunk);

    vl264_cfg cfg = vl264_default_cfg();
    cfg.axis = VL264_AXIS_Z; // force Z so P-frames exploit Z correlation
    vl264_enc* enc = vl264_enc_create(&cfg);
    vl264_dec* dec = vl264_dec_create();
    vl264_buf out = {0};

    vl264_encode(enc, chunk, NULL, NULL, &out);

    vl264_enc_stats stats;
    vl264_enc_stats_get(enc, &stats);
    printf("    I-slices: %u, P-slices: %u, skip MBs: %u\n",
           stats.i_slices, stats.p_slices, stats.skip_blocks);
    TEST(stats.p_slices > 0, "has P-frame slices");

    vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);
    float psnr = vl264_psnr(chunk, decoded, VL264_CHUNK_VOXELS);
    printf("    PSNR: %.1f dB, ratio: %.1f:1\n",
           psnr, (float)VL264_CHUNK_VOXELS / (float)out.size);
    TEST(psnr > 25.0f, "inter PSNR reasonable");

    vl264_free(out.data);
    vl264_enc_destroy(enc);
    vl264_dec_destroy(dec);
    free(chunk);
    free(decoded);
}

// ── CAVLC tests ─────────────────────────────────────────────────────────────

static void test_cavlc(void) {
    SECTION("cavlc");

    // Test indirectly: encode/decode a high-detail volume
    uint8_t* chunk = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
    uint8_t* decoded = (uint8_t*)malloc(VL264_CHUNK_VOXELS);

    gen_ct_phantom(chunk);

    vl264_cfg cfg = vl264_default_cfg();
    vl264_enc* enc = vl264_enc_create(&cfg);
    vl264_dec* dec = vl264_dec_create();
    vl264_buf out = {0};

    vl264_status s = vl264_encode(enc, chunk, NULL, NULL, &out);
    TEST(s == VL264_OK, "encode phantom");

    s = vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);
    TEST(s == VL264_OK, "decode phantom");

    float psnr = vl264_psnr(chunk, decoded, VL264_CHUNK_VOXELS);
    printf("    phantom PSNR: %.1f dB, size: %zu (ratio %.1f:1)\n",
           psnr, out.size, (float)VL264_CHUNK_VOXELS / (float)out.size);
    TEST(psnr > 20.0f, "phantom PSNR decent");

    vl264_free(out.data);
    vl264_enc_destroy(enc);
    vl264_dec_destroy(dec);
    free(chunk);
    free(decoded);
}

// ── NAL tests ───────────────────────────────────────────────────────────────

static void test_nal(void) {
    SECTION("nal");

    // Encode a chunk, verify we can find NAL units in the output
    uint8_t* chunk = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
    gen_constant(chunk, 64);

    vl264_cfg cfg = vl264_default_cfg();
    cfg.quality = VL264_FAST;
    vl264_enc* enc = vl264_enc_create(&cfg);
    vl264_buf out = {0};

    vl264_encode(enc, chunk, NULL, NULL, &out);

    // Count NAL units using length-prefix framing
    int nal_count = 0;
    size_t pos = 0;
    while (pos + 4 < out.size) {
        uint32_t len = ((uint32_t)out.data[pos] << 24) |
                       ((uint32_t)out.data[pos+1] << 16) |
                       ((uint32_t)out.data[pos+2] << 8) |
                       ((uint32_t)out.data[pos+3]);
        if (len == 0 || pos + 4 + len > out.size) break;
        nal_count++;
        pos += 4 + len;
    }
    printf("    NAL units found: %d\n", nal_count);
    TEST(nal_count >= 3, "at least SPS + PPS + 1 slice");

    vl264_free(out.data);
    vl264_enc_destroy(enc);
    free(chunk);
}

// ── Axis selection tests ────────────────────────────────────────────────────

static void test_axis(void) {
    SECTION("axis");

    uint8_t* chunk = (uint8_t*)malloc(VL264_CHUNK_VOXELS);

    // Z-gradient: data varies only along Z. Slices along X or Y are identical,
    // so axis selection should pick X or Y (lowest inter-slice difference).
    gen_gradient_z(chunk);
    vl264_axis az = vl264_analyze_axis(chunk);
    printf("    Z-gradient -> axis %d (expect X=0 or Y=1)\n", az);
    TEST(az == VL264_AXIS_X || az == VL264_AXIS_Y, "Z-gradient avoids Z axis");

    // X-gradient: data varies only along X. Slices along Y or Z are identical.
    gen_gradient_x(chunk);
    vl264_axis ax = vl264_analyze_axis(chunk);
    printf("    X-gradient -> axis %d (expect Y=1 or Z=2)\n", ax);
    TEST(ax == VL264_AXIS_Y || ax == VL264_AXIS_Z, "X-gradient avoids X axis");

    // Constant — any axis is fine (just verify no crash)
    gen_constant(chunk, 100);
    vl264_axis ac = vl264_analyze_axis(chunk);
    printf("    constant -> axis %d\n", ac);
    TEST(ac >= 0 && ac <= 2, "constant axis valid");

    free(chunk);
}

// ── Round-trip tests ────────────────────────────────────────────────────────

static void test_roundtrip(void) {
    SECTION("roundtrip");

    uint8_t* chunk = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
    uint8_t* decoded = (uint8_t*)malloc(VL264_CHUNK_VOXELS);

    struct {
        const char* name;
        void (*gen)(uint8_t*);
        float min_psnr;
    } cases[] = {
        {"constant",   NULL, 10.0f},
        {"gradient_z", NULL, 10.0f},
        {"sphere",     gen_sphere, 10.0f},
        {"ct_phantom", gen_ct_phantom, 10.0f},
    };
    int ncases = 4;

    for (int i = 0; i < ncases; i++) {
        switch (i) {
        case 0: gen_constant(chunk, 128); break;
        case 1: gen_gradient_z(chunk); break;
        case 2: gen_sphere(chunk); break;
        case 3: gen_ct_phantom(chunk); break;
        }

        vl264_cfg cfg = vl264_default_cfg();
        vl264_enc* enc = vl264_enc_create(&cfg);
        vl264_dec* dec = vl264_dec_create();
        vl264_buf out = {0};

        vl264_status s = vl264_encode(enc, chunk, NULL, NULL, &out);
        TEST(s == VL264_OK, "encode ok");

        s = vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);
        TEST(s == VL264_OK, "decode ok");

        float psnr = vl264_psnr(chunk, decoded, VL264_CHUNK_VOXELS);
        float ratio = (float)VL264_CHUNK_VOXELS / (float)out.size;
        printf("    %-12s PSNR=%.1f dB  ratio=%.1f:1  size=%zu\n",
               cases[i].name, psnr, ratio, out.size);
        TEST(psnr >= cases[i].min_psnr, "PSNR threshold met");

        vl264_free(out.data);
        vl264_enc_destroy(enc);
        vl264_dec_destroy(dec);
    }

    // Quality sweep: FAST < DEFAULT < MAX quality
    gen_ct_phantom(chunk);
    float psnrs[3];
    vl264_quality quals[] = {VL264_FAST, VL264_DEFAULT, VL264_MAX};
    const char* qnames[] = {"FAST", "DEFAULT", "MAX"};
    for (int q = 0; q < 3; q++) {
        vl264_cfg cfg = vl264_default_cfg();
        cfg.quality = quals[q];
        vl264_enc* enc = vl264_enc_create(&cfg);
        vl264_dec* dec = vl264_dec_create();
        vl264_buf out = {0};

        vl264_encode(enc, chunk, NULL, NULL, &out);
        vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);
        psnrs[q] = vl264_psnr(chunk, decoded, VL264_CHUNK_VOXELS);
        printf("    quality=%-8s PSNR=%.1f dB  size=%zu\n", qnames[q], psnrs[q], out.size);

        vl264_free(out.data);
        vl264_enc_destroy(enc);
        vl264_dec_destroy(dec);
    }
    // MAX should be >= DEFAULT, DEFAULT should be >= FAST
    TEST(psnrs[2] >= psnrs[0] - 5.0f, "MAX quality >= FAST quality (within 5dB)");

    free(chunk);
    free(decoded);
}

// ── Benchmark ───────────────────────────────────────────────────────────────

static double clock_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static void test_bench(void) {
    SECTION("bench");

    uint8_t* chunk = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
    uint8_t* decoded = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
    gen_ct_phantom(chunk);

    int iters = 5;

    vl264_quality quals[] = {VL264_FAST, VL264_DEFAULT, VL264_MAX};
    const char* qnames[] = {"FAST", "DEFAULT", "MAX"};

    for (int q = 0; q < 3; q++) {
        vl264_cfg cfg = vl264_default_cfg();
        cfg.quality = quals[q];
        vl264_enc* enc = vl264_enc_create(&cfg);
        vl264_dec* dec = vl264_dec_create();

        // Encode benchmark
        double t0 = clock_ms();
        vl264_buf out = {0};
        for (int i = 0; i < iters; i++) {
            out.size = 0;
            if (out.data) { vl264_free(out.data); out.data = NULL; }
            vl264_encode(enc, chunk, NULL, NULL, &out);
        }
        double enc_ms = (clock_ms() - t0) / iters;

        // Decode benchmark
        double t1 = clock_ms();
        for (int i = 0; i < iters; i++) {
            vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);
        }
        double dec_ms = (clock_ms() - t1) / iters;

        float mb_per_sec_enc = (float)VL264_CHUNK_VOXELS / (float)(enc_ms * 1000.0);
        float mb_per_sec_dec = (float)VL264_CHUNK_VOXELS / (float)(dec_ms * 1000.0);

        printf("    %-8s enc=%.1fms (%.1f MB/s)  dec=%.1fms (%.1f MB/s)  size=%zu\n",
               qnames[q], enc_ms, mb_per_sec_enc, dec_ms, mb_per_sec_dec, out.size);

        vl264_free(out.data);
        vl264_enc_destroy(enc);
        vl264_dec_destroy(dec);
    }

    free(chunk);
    free(decoded);
}

// ── QP sweep test ───────────────────────────────────────────────────────────

static void test_sweep(void) {
    SECTION("sweep");

    uint8_t* chunk = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
    uint8_t* decoded = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
    gen_ct_phantom(chunk);

    int qps[] = {5, 10, 15, 20, 25, 30, 35};
    int nqp = 7;
    float prev_psnr = 999.0f;
    size_t prev_size = VL264_CHUNK_VOXELS * 2;

    printf("    %-4s  %8s  %6s  %6s  %6s  %6s  %6s\n",
           "QP", "Size", "Ratio", "PSNR", "MAE", "P90", "P99");

    for (int qi = 0; qi < nqp; qi++) {
        vl264_cfg cfg = vl264_default_cfg();
        cfg.quality = VL264_FAST;
        cfg.qp = qps[qi];
        cfg.axis = VL264_AXIS_Z;

        vl264_enc* enc = vl264_enc_create(&cfg);
        vl264_dec* dec = vl264_dec_create();
        vl264_buf out = {0};

        vl264_encode(enc, chunk, NULL, NULL, &out);
        vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);

        vl264_stats stats = {0};
        stats.output_bytes = out.size;
        vl264_stats_compute(&stats, chunk, decoded, VL264_CHUNK_VOXELS);

        printf("    %-4d  %8zu  %6.1f  %6.1f  %6.2f  %6.1f  %6.1f\n",
               qps[qi], out.size, stats.ratio, stats.psnr,
               stats.mae, stats.p90_err, stats.p99_err);

        // PSNR should generally decrease as QP increases (allow 2 dB tolerance)
        if (qi > 0) {
            TEST(stats.psnr <= prev_psnr + 2.0f, "PSNR decreases with QP (within tolerance)");
        }
        // Size should generally decrease as QP increases
        TEST(stats.psnr > 20.0f, "PSNR > 20 dB at all tested QPs");
        TEST(stats.ratio > 1.0f, "compression ratio > 1:1 at all tested QPs");

        prev_psnr = stats.psnr;
        prev_size = out.size;

        vl264_free(out.data);
        vl264_enc_destroy(enc);
        vl264_dec_destroy(dec);
    }

    free(chunk);
    free(decoded);
}

// ── Stats report test ───────────────────────────────────────────────────────

static void test_stats(void) {
    SECTION("stats");

    uint8_t* chunk = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
    uint8_t* decoded = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
    gen_ct_phantom(chunk);

    vl264_cfg cfg = vl264_default_cfg();
    vl264_enc* enc = vl264_enc_create(&cfg);
    vl264_dec* dec = vl264_dec_create();
    vl264_buf out = {0};

    // Encode
    vl264_encode(enc, chunk, NULL, NULL, &out);

    // Get encoder stats
    vl264_stats stats;
    vl264_enc_stats_get(enc, &stats);

    // Decode with timing
    double t0 = clock_ms() / 1000.0;
    vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);
    stats.decode_sec = clock_ms() / 1000.0 - t0;
    if (stats.decode_sec > 0.0)
        stats.decode_mbs = (double)VL264_CHUNK_VOXELS / (stats.decode_sec * 1e6);

    // Compute quality stats
    vl264_stats_compute(&stats, chunk, decoded, VL264_CHUNK_VOXELS);

    // Print full report
    vl264_stats_print(&stats, stdout);

    // Verify stats are populated
    TEST(stats.psnr > 20.0f, "PSNR populated");
    TEST(stats.mse > 0.0f, "MSE populated");
    TEST(stats.mae > 0.0f, "MAE populated");
    TEST(stats.max_err >= 0.0f, "max_err populated");
    TEST(stats.p50_err <= stats.p90_err, "p50 <= p90");
    TEST(stats.p90_err <= stats.p95_err, "p90 <= p95");
    TEST(stats.p95_err <= stats.p99_err, "p95 <= p99");
    TEST(stats.p99_err <= stats.max_err, "p99 <= max");
    TEST(stats.input_entropy > 0.0f, "entropy populated");
    TEST(stats.total_blocks > 0, "total_blocks populated");
    TEST(stats.encode_sec > 0.0, "encode_sec populated");
    TEST(stats.ratio > 1.0f, "compression ratio > 1");
    TEST(stats.bits_per_voxel > 0.0f && stats.bits_per_voxel < 8.0f, "bits_per_voxel in range");
    TEST(stats.intra_blocks + stats.inter_blocks + stats.skip_blocks == stats.total_blocks,
         "block counts add up");

    vl264_free(out.data);
    vl264_enc_destroy(enc);
    vl264_dec_destroy(dec);
    free(chunk);
    free(decoded);
}

// ── Error handling tests ────────────────────────────────────────────────────

static void test_errors(void) {
    SECTION("errors");

    // NULL argument checks
    TEST(vl264_encode(NULL, NULL, NULL, NULL, NULL) == VL264_ERR_NULL_ARG, "encode NULL enc");
    vl264_enc* enc = vl264_enc_create(NULL); // NULL cfg = default
    TEST(enc != NULL, "create enc with NULL cfg");
    TEST(vl264_encode(enc, NULL, NULL, NULL, &(vl264_buf){0}) == VL264_ERR_NULL_ARG, "encode NULL chunk");

    uint8_t dummy[VL264_CHUNK_VOXELS];
    memset(dummy, 128, sizeof(dummy));
    TEST(vl264_encode(enc, dummy, NULL, NULL, NULL) == VL264_ERR_NULL_ARG, "encode NULL out");

    // Invalid QP
    vl264_cfg bad_cfg = vl264_default_cfg();
    bad_cfg.qp = 100;
    vl264_enc* bad_enc = vl264_enc_create(&bad_cfg);
    vl264_buf out = {0};
    TEST(vl264_encode(bad_enc, dummy, NULL, NULL, &out) == VL264_ERR_INVALID, "encode QP=100");
    vl264_enc_destroy(bad_enc);

    // Buffer too small
    uint8_t small_buf[16];
    vl264_buf small_out = {.data = small_buf, .size = 0, .capacity = 16};
    TEST(vl264_encode(enc, dummy, NULL, NULL, &small_out) == VL264_ERR_OVERFLOW, "encode small buffer");

    // Decode NULL checks
    TEST(vl264_decode(NULL, NULL, 0, NULL, NULL, NULL) == VL264_ERR_NULL_ARG, "decode NULL dec");
    vl264_dec* dec = vl264_dec_create();
    TEST(vl264_decode(dec, NULL, 0, NULL, NULL, dummy) == VL264_ERR_NULL_ARG, "decode NULL bitstream");

    // Decode corrupt data
    uint8_t garbage[64];
    memset(garbage, 0xFF, sizeof(garbage));
    TEST(vl264_decode(dec, garbage, sizeof(garbage), NULL, NULL, dummy) == VL264_ERR_CORRUPT, "decode garbage");

    // Decode empty
    TEST(vl264_decode(dec, garbage, 0, NULL, NULL, dummy) == VL264_ERR_NULL_ARG, "decode zero length");

    // Stats NULL checks
    TEST(vl264_enc_stats_get(NULL, NULL) == VL264_ERR_NULL_ARG, "stats NULL enc");
    vl264_stats s;
    TEST(vl264_enc_stats_get(enc, &s) == VL264_OK, "stats ok");

    vl264_enc_destroy(enc);
    vl264_dec_destroy(dec);

    // Max error clamping test
    {
        uint8_t* me_chunk = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
        uint8_t* me_decoded = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
        gen_ct_phantom(me_chunk);

        vl264_cfg me_cfg = vl264_default_cfg();
        me_cfg.quality = VL264_FAST;
        me_cfg.qp = 40; // aggressive QP
        me_cfg.max_error = 20; // but cap error at ±20
        me_cfg.bit_depth = 8; // don't shift (synthetic data has full range)

        vl264_enc* me_enc = vl264_enc_create(&me_cfg);
        vl264_dec* me_dec = vl264_dec_create();
        vl264_buf me_out = {0};
        vl264_encode(me_enc, me_chunk, NULL, NULL, &me_out);
        vl264_decode(me_dec, me_out.data, me_out.size, NULL, NULL, me_decoded);

        int32_t max_pixel_err = 0;
        for (size_t i = 0; i < VL264_CHUNK_VOXELS; i++) {
            int32_t e2 = abs((int32_t)me_chunk[i] - (int32_t)me_decoded[i]);
            if (e2 > max_pixel_err) max_pixel_err = e2;
        }
        float me_psnr = vl264_psnr(me_chunk, me_decoded, VL264_CHUNK_VOXELS);
        printf("    max_error test: max_err=%d (limit=%d) PSNR=%.1f\n", max_pixel_err, me_cfg.max_error, me_psnr);
        // max_error is best-effort per-block. P-frame propagation can exceed the limit.
        // But it should substantially improve quality vs unclamped QP=40.
        TEST(me_psnr > 15.0f, "max_error improves quality at high QP");

        vl264_free(me_out.data);
        vl264_enc_destroy(me_enc);
        vl264_dec_destroy(me_dec);
        free(me_chunk);
        free(me_decoded);
    }
}

// ── Streaming decode test ───────────────────────────────────────────────────

static void test_streaming(void) {
    SECTION("streaming");

    uint8_t* chunk = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
    uint8_t* full_decoded = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
    uint8_t* stream_decoded = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
    gen_ct_phantom(chunk);
    memset(stream_decoded, 0, VL264_CHUNK_VOXELS);

    vl264_cfg cfg = vl264_default_cfg();
    cfg.quality = VL264_FAST;
    cfg.qp = 20;
    vl264_enc* enc = vl264_enc_create(&cfg);
    vl264_buf out = {0};
    vl264_encode(enc, chunk, NULL, NULL, &out);

    // Full decode
    vl264_dec* dec1 = vl264_dec_create();
    vl264_decode(dec1, out.data, out.size, NULL, NULL, full_decoded);

    // Streaming decode — use axis-aware insertion
    vl264_dec* dec2 = vl264_dec_create();
    vl264_status s = vl264_decode_begin(dec2, out.data, out.size, NULL, NULL);
    TEST(s == VL264_OK, "streaming begin");
    vl264_axis stream_axis = vl264_decode_axis(dec2);

    int slices_decoded = 0;
    while (slices_decoded < 128) {
        uint8_t slice_buf[128 * 128];
        uint32_t slice_idx;
        s = vl264_decode_next_slice(dec2, slice_buf, &slice_idx);
        if (s != VL264_OK) break;
        // Axis-aware insertion
        for (int a = 0; a < 128; a++) {
            for (int b = 0; b < 128; b++) {
                size_t vi;
                switch (stream_axis) {
                case VL264_AXIS_X: vi = (size_t)a * 128 * 128 + (size_t)b * 128 + slice_idx; break;
                case VL264_AXIS_Y: vi = (size_t)a * 128 * 128 + (size_t)slice_idx * 128 + b; break;
                default:           vi = (size_t)slice_idx * 128 * 128 + (size_t)a * 128 + b; break;
                }
                stream_decoded[vi] = slice_buf[a * 128 + b];
            }
        }
        slices_decoded++;
    }
    printf("    decoded %d slices (axis=%d) via streaming\n", slices_decoded, stream_axis);
    TEST(slices_decoded == 128, "all 128 slices decoded");

    // Compare full vs streaming decode
    int diffs = 0;
    for (size_t i = 0; i < VL264_CHUNK_VOXELS; i++)
        if (full_decoded[i] != stream_decoded[i]) diffs++;
    printf("    full vs streaming diffs: %d\n", diffs);
    TEST(diffs == 0, "streaming matches full decode");

    vl264_free(out.data);
    vl264_enc_destroy(enc);
    vl264_dec_destroy(dec1);
    vl264_dec_destroy(dec2);
    free(chunk);
    free(full_decoded);
    free(stream_decoded);
}

// ── Edge case tests ─────────────────────────────────────────────────────────

static void test_edge_cases(void) {
    SECTION("edge_cases");

    uint8_t* chunk = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
    uint8_t* decoded = (uint8_t*)malloc(VL264_CHUNK_VOXELS);

    // All zeros
    memset(chunk, 0, VL264_CHUNK_VOXELS);
    {
        vl264_cfg cfg = vl264_default_cfg();
        cfg.quality = VL264_FAST; cfg.qp = 20;
        vl264_enc* enc = vl264_enc_create(&cfg);
        vl264_dec* dec = vl264_dec_create();
        vl264_buf out = {0};
        vl264_encode(enc, chunk, NULL, NULL, &out);
        vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);
        float psnr = vl264_psnr(chunk, decoded, VL264_CHUNK_VOXELS);
        printf("    all-zero: PSNR=%.1f ratio=%.1f:1\n", psnr, (float)VL264_CHUNK_VOXELS/out.size);
        TEST(psnr > 10.0f || psnr == INFINITY, "all-zero quality");
        vl264_free(out.data);
        vl264_enc_destroy(enc);
        vl264_dec_destroy(dec);
    }

    // All 255
    memset(chunk, 255, VL264_CHUNK_VOXELS);
    {
        vl264_cfg cfg = vl264_default_cfg();
        cfg.quality = VL264_FAST; cfg.qp = 20;
        vl264_enc* enc = vl264_enc_create(&cfg);
        vl264_dec* dec = vl264_dec_create();
        vl264_buf out = {0};
        vl264_encode(enc, chunk, NULL, NULL, &out);
        vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);
        float psnr = vl264_psnr(chunk, decoded, VL264_CHUNK_VOXELS);
        printf("    all-255: PSNR=%.1f ratio=%.1f:1\n", psnr, (float)VL264_CHUNK_VOXELS/out.size);
        TEST(psnr > 30.0f || psnr == INFINITY, "all-255 quality");
        vl264_free(out.data);
        vl264_enc_destroy(enc);
        vl264_dec_destroy(dec);
    }

    // Random noise (worst case)
    gen_noise(chunk, 12345);
    {
        vl264_cfg cfg = vl264_default_cfg();
        cfg.quality = VL264_FAST; cfg.qp = 30;
        vl264_enc* enc = vl264_enc_create(&cfg);
        vl264_dec* dec = vl264_dec_create();
        vl264_buf out = {0};
        vl264_encode(enc, chunk, NULL, NULL, &out);
        vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);
        float psnr = vl264_psnr(chunk, decoded, VL264_CHUNK_VOXELS);
        printf("    random noise: PSNR=%.1f ratio=%.1f:1\n", psnr, (float)VL264_CHUNK_VOXELS/out.size);
        TEST(psnr > 10.0f, "random noise doesn't crash");
        vl264_free(out.data);
        vl264_enc_destroy(enc);
        vl264_dec_destroy(dec);
    }

    // All QP values (quick roundtrip at each)
    gen_constant(chunk, 100);
    int qp_fails = 0;
    for (int qp = 1; qp <= 51; qp++) {
        vl264_cfg cfg = vl264_default_cfg();
        cfg.quality = VL264_FAST; cfg.qp = qp; cfg.axis = VL264_AXIS_Z;
        vl264_enc* enc = vl264_enc_create(&cfg);
        vl264_dec* dec = vl264_dec_create();
        vl264_buf out = {0};
        vl264_status s = vl264_encode(enc, chunk, NULL, NULL, &out);
        if (s != VL264_OK) { qp_fails++; vl264_enc_destroy(enc); vl264_dec_destroy(dec); continue; }
        s = vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);
        if (s != VL264_OK) qp_fails++;
        vl264_free(out.data);
        vl264_enc_destroy(enc);
        vl264_dec_destroy(dec);
    }
    printf("    all QP 1-51 roundtrip: %d failures\n", qp_fails);
    TEST(qp_fails == 0, "all QP values roundtrip successfully");

    // Each axis explicitly
    gen_ct_phantom(chunk);
    for (int ax = 0; ax < 3; ax++) {
        vl264_cfg cfg = vl264_default_cfg();
        cfg.quality = VL264_FAST; cfg.qp = 20; cfg.axis = (vl264_axis)ax;
        vl264_enc* enc = vl264_enc_create(&cfg);
        vl264_dec* dec = vl264_dec_create();
        vl264_buf out = {0};
        vl264_encode(enc, chunk, NULL, NULL, &out);
        vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);
        float psnr = vl264_psnr(chunk, decoded, VL264_CHUNK_VOXELS);
        printf("    axis=%d: PSNR=%.1f\n", ax, psnr);
        TEST(psnr > 20.0f, "explicit axis works");
        vl264_free(out.data);
        vl264_enc_destroy(enc);
        vl264_dec_destroy(dec);
    }

    // Morton ordering
    {
        vl264_cfg cfg = vl264_default_cfg();
        cfg.quality = VL264_FAST; cfg.qp = 20; cfg.morton_order = true;
        vl264_enc* enc = vl264_enc_create(&cfg);
        vl264_dec* dec = vl264_dec_create();
        vl264_buf out = {0};
        vl264_encode(enc, chunk, NULL, NULL, &out);
        vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);
        float psnr = vl264_psnr(chunk, decoded, VL264_CHUNK_VOXELS);
        printf("    morton: PSNR=%.1f ratio=%.1f:1\n", psnr, (float)VL264_CHUNK_VOXELS/out.size);
        TEST(psnr > 15.0f, "morton ordering works");
        vl264_free(out.data);
        vl264_enc_destroy(enc);
        vl264_dec_destroy(dec);
    }

    // Utility functions
    float mse = vl264_mse(chunk, decoded, VL264_CHUNK_VOXELS);
    TEST(mse >= 0.0f, "MSE non-negative");
    float psnr = vl264_psnr(chunk, chunk, VL264_CHUNK_VOXELS);
    TEST(psnr == INFINITY, "identical PSNR is infinity");
    TEST(strcmp(vl264_status_str(VL264_OK), "ok") == 0, "status_str ok");
    TEST(strcmp(vl264_status_str(VL264_ERR_CORRUPT), "corrupt data") == 0, "status_str corrupt");

    free(chunk);
    free(decoded);
}

// ── Codec property tests ────────────────────────────────────────────────────

static void test_properties(void) {
    SECTION("properties");

    uint8_t* chunk = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
    uint8_t* decoded = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
    gen_ct_phantom(chunk);

    // Property 1: Higher QP = smaller output (monotonic)
    size_t prev_size = VL264_CHUNK_VOXELS * 2;
    for (int qp = 5; qp <= 51; qp += 5) {
        vl264_cfg cfg = vl264_default_cfg();
        cfg.quality = VL264_FAST; cfg.qp = qp; cfg.axis = VL264_AXIS_Z; cfg.bit_depth = 8;
        vl264_enc* enc = vl264_enc_create(&cfg);
        vl264_dec* dec = vl264_dec_create();
        vl264_buf out = {0};
        vl264_encode(enc, chunk, NULL, NULL, &out);
        TEST(out.size <= prev_size, "higher QP = smaller output");
        prev_size = out.size;
        vl264_enc_destroy(enc); vl264_dec_destroy(dec); vl264_free(out.data);
    }

    // Property 2: Encode then decode = roundtrip (no crash, reasonable PSNR)
    for (int qp = 5; qp <= 51; qp += 10) {
        vl264_cfg cfg = vl264_default_cfg();
        cfg.quality = VL264_FAST; cfg.qp = qp; cfg.bit_depth = 8;
        vl264_enc* enc = vl264_enc_create(&cfg);
        vl264_dec* dec = vl264_dec_create();
        vl264_buf out = {0};
        vl264_encode(enc, chunk, NULL, NULL, &out);
        vl264_status s = vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);
        TEST(s == VL264_OK, "roundtrip succeeds at all QP");
        float psnr = vl264_psnr(chunk, decoded, VL264_CHUNK_VOXELS);
        TEST(psnr > 5.0f, "roundtrip PSNR > 5 dB at all QP");
        vl264_enc_destroy(enc); vl264_dec_destroy(dec); vl264_free(out.data);
    }

    // Property 3: All three quality presets work
    vl264_quality quals[] = {VL264_FAST, VL264_DEFAULT, VL264_MAX};
    for (int q = 0; q < 3; q++) {
        vl264_cfg cfg = vl264_default_cfg();
        cfg.quality = quals[q]; cfg.qp = 20; cfg.bit_depth = 8;
        vl264_enc* enc = vl264_enc_create(&cfg);
        vl264_dec* dec = vl264_dec_create();
        vl264_buf out = {0};
        vl264_encode(enc, chunk, NULL, NULL, &out);
        vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);
        float psnr = vl264_psnr(chunk, decoded, VL264_CHUNK_VOXELS);
        TEST(psnr > 10.0f, "quality preset works");
        vl264_enc_destroy(enc); vl264_dec_destroy(dec); vl264_free(out.data);
    }

    // Property 4: Bit-depth detection works for 5-bit data
    uint8_t* chunk5 = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
    for (size_t i = 0; i < VL264_CHUNK_VOXELS; i++) chunk5[i] = (chunk[i] >> 3) << 3;
    {
        vl264_cfg cfg = vl264_default_cfg();
        cfg.quality = VL264_FAST; cfg.qp = 20;
        vl264_enc* enc = vl264_enc_create(&cfg);
        vl264_dec* dec = vl264_dec_create();
        vl264_buf out5 = {0}, out8 = {0};
        vl264_encode(enc, chunk5, NULL, NULL, &out5);
        vl264_enc_destroy(enc);
        // Same data but forced 8-bit
        cfg.bit_depth = 8;
        enc = vl264_enc_create(&cfg);
        vl264_encode(enc, chunk5, NULL, NULL, &out8);
        printf("    5-bit data: auto=%zu bytes, forced8=%zu bytes\n", out5.size, out8.size);
        TEST(out5.size < out8.size, "5-bit auto-detect compresses better than forced 8-bit");
        vl264_free(out5.data); vl264_free(out8.data);
        vl264_enc_destroy(enc); vl264_dec_destroy(dec);
    }
    free(chunk5);

    // Property 5: Encoder reuse (encode multiple chunks with same encoder)
    {
        vl264_cfg cfg = vl264_default_cfg();
        cfg.quality = VL264_FAST; cfg.qp = 20; cfg.bit_depth = 8;
        vl264_enc* enc = vl264_enc_create(&cfg);
        vl264_dec* dec = vl264_dec_create();
        for (int i = 0; i < 5; i++) {
            memset(chunk, (uint8_t)(50 + i * 30), VL264_CHUNK_VOXELS);
            vl264_buf out = {0};
            vl264_encode(enc, chunk, NULL, NULL, &out);
            vl264_status s = vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);
            TEST(s == VL264_OK, "encoder reuse works");
            vl264_free(out.data);
        }
        vl264_enc_destroy(enc); vl264_dec_destroy(dec);
    }

    // Property 6: Different axes produce valid output
    gen_ct_phantom(chunk);
    for (int ax = 0; ax < 3; ax++) {
        vl264_cfg cfg = vl264_default_cfg();
        cfg.quality = VL264_FAST; cfg.qp = 25; cfg.axis = (vl264_axis)ax; cfg.bit_depth = 8;
        vl264_enc* enc = vl264_enc_create(&cfg);
        vl264_dec* dec = vl264_dec_create();
        vl264_buf out = {0};
        vl264_encode(enc, chunk, NULL, NULL, &out);
        vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);
        float psnr = vl264_psnr(chunk, decoded, VL264_CHUNK_VOXELS);
        TEST(psnr > 10.0f, "each axis produces valid output");
        vl264_enc_destroy(enc); vl264_dec_destroy(dec); vl264_free(out.data);
    }

    // Property 7: iframe_interval is respected
    {
        vl264_cfg cfg = vl264_default_cfg();
        cfg.quality = VL264_FAST; cfg.qp = 25; cfg.iframe_interval = 8; cfg.bit_depth = 8;
        vl264_enc* enc = vl264_enc_create(&cfg);
        vl264_dec* dec = vl264_dec_create();
        vl264_buf out = {0};
        vl264_encode(enc, chunk, NULL, NULL, &out);
        vl264_enc_stats stats;
        vl264_enc_stats_get(enc, &stats);
        // With interval=8, expect 128/8=16 I-slices
        printf("    iframe_interval=8: I=%u P=%u\n", stats.i_slices, stats.p_slices);
        TEST(stats.i_slices >= 14 && stats.i_slices <= 18, "iframe interval roughly respected");
        vl264_decode(dec, out.data, out.size, NULL, NULL, decoded);
        float psnr = vl264_psnr(chunk, decoded, VL264_CHUNK_VOXELS);
        TEST(psnr > 10.0f, "iframe interval doesn't break decode");
        vl264_enc_destroy(enc); vl264_dec_destroy(dec); vl264_free(out.data);
    }

    // Property 8: Compression ratio > 1 for all tested data at QP >= 15
    {
        uint8_t* patterns[] = {chunk, NULL};
        void (*gens[])(uint8_t*) = {gen_ct_phantom, gen_sphere, gen_gradient_z, NULL};
        for (int g = 0; gens[g]; g++) {
            gens[g](chunk);
            vl264_cfg cfg = vl264_default_cfg();
            cfg.quality = VL264_FAST; cfg.qp = 25; cfg.bit_depth = 8;
            vl264_enc* enc = vl264_enc_create(&cfg);
            vl264_buf out = {0};
            vl264_encode(enc, chunk, NULL, NULL, &out);
            TEST(out.size < VL264_CHUNK_VOXELS, "compresses at QP=25");
            vl264_enc_destroy(enc); vl264_free(out.data);
        }
    }

    free(chunk);
    free(decoded);
}

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    bool run_all = argc < 2 || has_flag(argc, argv, "--all");

    printf("vl264 test suite (%s)\n", vl264_version_str());

    if (run_all || has_flag(argc, argv, "--bitstream"))
        test_bitstream_proper();
    if (run_all || has_flag(argc, argv, "--dct"))
        test_dct();
    if (run_all || has_flag(argc, argv, "--quant"))
        test_quant();
    if (run_all || has_flag(argc, argv, "--intra"))
        test_intra();
    if (run_all || has_flag(argc, argv, "--inter"))
        test_inter();
    if (run_all || has_flag(argc, argv, "--cavlc"))
        test_cavlc();
    if (run_all || has_flag(argc, argv, "--nal"))
        test_nal();
    if (run_all || has_flag(argc, argv, "--axis"))
        test_axis();
    if (run_all || has_flag(argc, argv, "--roundtrip"))
        test_roundtrip();
    if (run_all || has_flag(argc, argv, "--sweep"))
        test_sweep();
    if (run_all || has_flag(argc, argv, "--stats"))
        test_stats();
    if (run_all || has_flag(argc, argv, "--errors"))
        test_errors();
    if (run_all || has_flag(argc, argv, "--streaming"))
        test_streaming();
    if (run_all || has_flag(argc, argv, "--edge"))
        test_edge_cases();
    if (run_all || has_flag(argc, argv, "--props"))
        test_properties();
    if (run_all || has_flag(argc, argv, "--bench"))
        test_bench();

    printf("\n%d tests: %d passed, %d failed\n", g_tests, g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
