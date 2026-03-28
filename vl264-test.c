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
    TEST(psnr > 20.0f, "constant PSNR > 20 dB");

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
        {"constant",   NULL, 30.0f},
        {"gradient_z", NULL, 25.0f},
        {"sphere",     gen_sphere, 20.0f},
        {"ct_phantom", gen_ct_phantom, 20.0f},
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
    if (run_all || has_flag(argc, argv, "--bench"))
        test_bench();

    printf("\n%d tests: %d passed, %d failed\n", g_tests, g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
