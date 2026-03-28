// vl264.c — Volume Librarian 264 implementation
// Pure C23 codec for 128x128x128 u8 volumetric CT chunks.
#define _POSIX_C_SOURCE 199309L
#include "vl264.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ═════════════════════════════════════════════════════════════════════════════
// Section 1: Preamble — macros, internal types
// ═════════════════════════════════════════════════════════════════════════════

#define VL264_INTERNAL    static
#define VL264_RESTRICT    restrict
#define VL264_ALIGNED(n)  _Alignas(n)

// Compiler hints
#if defined(__clang__) || defined(__GNUC__)
#define VL264_LIKELY(x)     __builtin_expect(!!(x), 1)
#define VL264_UNLIKELY(x)   __builtin_expect(!!(x), 0)
#define VL264_HOT            __attribute__((hot))
#define VL264_COLD           __attribute__((cold))
#define VL264_PURE           __attribute__((pure))
#define VL264_CONST_FN       __attribute__((const))
#define VL264_FLATTEN        __attribute__((flatten))
#define VL264_ALWAYS_INLINE  __attribute__((always_inline)) static inline
#define VL264_NOINLINE       __attribute__((noinline))
#define VL264_PREFETCH(p)    __builtin_prefetch(p)
#else
#define VL264_LIKELY(x)     (x)
#define VL264_UNLIKELY(x)   (x)
#define VL264_HOT
#define VL264_COLD
#define VL264_PURE
#define VL264_CONST_FN
#define VL264_FLATTEN
#define VL264_ALWAYS_INLINE  static inline
#define VL264_NOINLINE
#define VL264_PREFETCH(p)
#endif

#define VL264_MIN(a, b) ((a) < (b) ? (a) : (b))
#define VL264_MAX(a, b) ((a) > (b) ? (a) : (b))
#define VL264_CLAMP(x, lo, hi) VL264_MIN(VL264_MAX((x), (lo)), (hi))

// Safe shift: avoids UB when shift >= 32
VL264_ALWAYS_INLINE VL264_CONST_FN uint32_t shl32(uint32_t v, int32_t n) { return n >= 32 ? 0 : n <= 0 ? v : v << n; }
VL264_ALWAYS_INLINE VL264_CONST_FN uint32_t shr32(uint32_t v, int32_t n) { return n >= 32 ? 0 : n <= 0 ? v : v >> n; }

#define DIM   128
#define BDIM  4    // block dimension
#define BSTRIDE (DIM / BDIM)  // 32 blocks per row

// NAL unit types
#define NAL_SPS       7
#define NAL_PPS       8
#define NAL_SLICE_IDR 5
#define NAL_SLICE_P   1
#define NAL_SEI       6

// MB types
#define MB_TYPE_I4x4  0
#define MB_TYPE_P     1
#define MB_TYPE_SKIP  2

// Intra 4x4 prediction modes
#define INTRA_VERT      0
#define INTRA_HORIZ     1
#define INTRA_DC        2
#define INTRA_DDL       3
#define INTRA_DDR       4
#define INTRA_VR        5
#define INTRA_HD        6
#define INTRA_VL        7
#define INTRA_HU        8
#define INTRA_NUM_MODES 9

// Timing helper (needs POSIX clock_gettime)
VL264_INTERNAL double vl264_clock_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// ═════════════════════════════════════════════════════════════════════════════
// Section 2: Bitstream writer / reader
// ═════════════════════════════════════════════════════════════════════════════

typedef struct {
    uint8_t* buf;
    size_t   capacity;
    size_t   byte_pos;
    uint32_t cache;
    int32_t  bits_left; // bits free in cache (write from MSB down)
} bs_writer;

VL264_INTERNAL void bs_w_init(bs_writer* w, uint8_t* buf, size_t cap) {
    w->buf = buf;
    w->capacity = cap;
    w->byte_pos = 0;
    w->cache = 0;
    w->bits_left = 32;
}

VL264_INTERNAL void bs_w_flush_cache(bs_writer* w) {
    while (w->bits_left <= 24 && w->byte_pos < w->capacity) {
        w->buf[w->byte_pos++] = (uint8_t)(w->cache >> 24);
        w->cache <<= 8;
        w->bits_left += 8;
    }
}

VL264_INTERNAL VL264_HOT void bs_w_bits(bs_writer* w, uint32_t val, int32_t n) {
    if (VL264_UNLIKELY(n <= 0)) return;
    // Fast path: fits entirely in cache (common case)
    if (VL264_LIKELY(n <= w->bits_left && n < 32)) {
        w->cache |= (val & ((1u << n) - 1)) << (w->bits_left - n);
        w->bits_left -= n;
        if (w->bits_left <= 8) bs_w_flush_cache(w);
        return;
    }
    // Slow path: split across boundary
    if (n > 32) n = 32;
    uint32_t masked = (n >= 32) ? val : (val & (shl32(1u, n) - 1));
    while (n > 0) {
        int32_t write_n = n < w->bits_left ? n : w->bits_left;
        if (write_n <= 0) { bs_w_flush_cache(w); continue; }
        uint32_t bits = shr32(masked, n - write_n) & (shl32(1u, write_n) - 1);
        w->cache |= shl32(bits, w->bits_left - write_n);
        w->bits_left -= write_n;
        n -= write_n;
        if (w->bits_left <= 0) bs_w_flush_cache(w);
    }
}

VL264_INTERNAL void bs_w_bit1(bs_writer* w, uint32_t val) {
    bs_w_bits(w, val, 1);
}

// Unsigned exp-golomb (H.264 9.1)
VL264_INTERNAL void bs_w_ue(bs_writer* w, uint32_t val) {
    uint32_t v = val + 1;
    int32_t leading = 0;
    uint32_t tmp = v;
    while (tmp >>= 1) leading++;
    bs_w_bits(w, 0, leading);
    bs_w_bits(w, v, leading + 1);
}

// Signed exp-golomb
VL264_INTERNAL void bs_w_se(bs_writer* w, int32_t val) {
    if (val > 0)
        bs_w_ue(w, (uint32_t)(2 * val - 1));
    else
        bs_w_ue(w, (uint32_t)(-2 * val));
}

VL264_INTERNAL void bs_w_align(bs_writer* w) {
    int32_t rem = w->bits_left & 7;
    if (rem != 0 && rem != 8) {
        bs_w_bits(w, 0, rem);
    }
    bs_w_flush_cache(w);
}

VL264_INTERNAL void bs_w_flush(bs_writer* w) {
    while (w->bits_left < 32 && w->byte_pos < w->capacity) {
        w->buf[w->byte_pos++] = (uint8_t)(w->cache >> 24);
        w->cache <<= 8;
        w->bits_left += 8;
    }
}


// Reader
typedef struct {
    const uint8_t* buf;
    size_t         size;
    size_t         byte_pos;
    uint32_t       cache;
    int32_t        bits_left; // valid bits in cache (from MSB)
} bs_reader;

VL264_INTERNAL void bs_r_init(bs_reader* r, const uint8_t* buf, size_t size) {
    r->buf = buf;
    r->size = size;
    r->byte_pos = 0;
    r->cache = 0;
    r->bits_left = 0;
}

VL264_INTERNAL void bs_r_refill(bs_reader* r) {
    while (r->bits_left <= 24 && r->byte_pos < r->size) {
        r->cache |= (uint32_t)r->buf[r->byte_pos++] << (24 - r->bits_left);
        r->bits_left += 8;
    }
}

VL264_INTERNAL VL264_HOT uint32_t bs_r_bits(bs_reader* r, int32_t n) {
    if (n <= 0) return 0;
    bs_r_refill(r);
    // Fast path (n < 25 and enough bits — almost always true)
    if (n < 25 && n <= r->bits_left) {
        uint32_t val = r->cache >> (32 - n);
        r->cache <<= n;
        r->bits_left -= n;
        return val;
    }
    if (n > 32) n = 32;
    uint32_t val = shr32(r->cache, 32 - n);
    r->cache = shl32(r->cache, n);
    r->bits_left -= n;
    return val;
}

VL264_INTERNAL uint32_t bs_r_bit1(bs_reader* r) {
    return bs_r_bits(r, 1);
}

VL264_INTERNAL uint32_t bs_r_ue(bs_reader* r) {
    int32_t leading = 0;
    while (bs_r_bit1(r) == 0 && leading < 32) leading++;
    if (leading == 0) return 0;
    uint32_t suffix = bs_r_bits(r, leading);
    return (1u << leading) - 1 + suffix;
}

VL264_INTERNAL int32_t bs_r_se(bs_reader* r) {
    uint32_t v = bs_r_ue(r);
    if (v & 1) return (int32_t)((v + 1) / 2);
    else       return -(int32_t)(v / 2);
}

VL264_INTERNAL void bs_r_align(bs_reader* r) {
    int32_t skip = r->bits_left & 7;
    if (skip > 0) {
        r->cache <<= skip;
        r->bits_left -= skip;
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// Section 3: NAL framing (length-prefixed) ──────────────────────────────────────────────
// 2-byte big-endian length prefix (max 65535 bytes per NAL).
// Format: [2-byte length][1-byte nal_type][payload...]

#define NAL_LEN_BYTES 2
#define NAL_MAX_SIZE  65535

VL264_INTERNAL size_t find_nal_units(const uint8_t* data, size_t size,
                                      size_t* offsets, size_t* sizes, size_t max_nals) {
    size_t count = 0;
    size_t pos = 0;
    while (pos + NAL_LEN_BYTES < size && count < max_nals) {
        uint32_t len = ((uint32_t)data[pos] << 8) | (uint32_t)data[pos+1];
        pos += NAL_LEN_BYTES;
        if (len == 0 || pos + len > size) break;
        offsets[count] = pos;
        sizes[count] = (size_t)len;
        count++;
        pos += len;
    }
    return count;
}

// ═════════════════════════════════════════════════════════════════════════════
// Section 4: NAL / SPS / PPS / Slice headers
// ═════════════════════════════════════════════════════════════════════════════

// VL264 SPS — stored in bitstream
typedef struct {
    uint8_t  axis;          // encode axis (0,1,2)
    uint8_t  qp_base;       // base QP
    uint8_t  morton;        // 0=raster, 1=morton
    uint8_t  has_boundary;  // boundary prediction flag
    uint8_t  has_lod;       // LOD delta coding flag
    uint8_t  bit_shift;     // right-shift applied before encoding (0-5)
} vl264_sps_data;

// VL264 slice header
typedef struct {
    uint32_t slice_idx;     // spatial slice index
    uint8_t  is_idr;        // 1 = I-frame
    int8_t   qp_delta;      // QP delta from base
    uint32_t ref_idx;       // reference slice index (for P-frames)
} vl264_slice_hdr;

VL264_INTERNAL void write_sps(bs_writer* w, const vl264_sps_data* sps) {
    bs_w_ue(w, sps->axis);
    bs_w_ue(w, sps->qp_base);
    bs_w_bit1(w, sps->morton);
    bs_w_bit1(w, sps->has_boundary);
    bs_w_bit1(w, sps->has_lod);
    bs_w_bits(w, sps->bit_shift, 3); // 0-5 shift amount
    bs_w_align(w);
}

VL264_INTERNAL void read_sps(bs_reader* r, vl264_sps_data* sps) {
    sps->axis = (uint8_t)bs_r_ue(r);
    sps->qp_base = (uint8_t)bs_r_ue(r);
    sps->morton = (uint8_t)bs_r_bit1(r);
    sps->has_boundary = (uint8_t)bs_r_bit1(r);
    sps->has_lod = (uint8_t)bs_r_bit1(r);
    sps->bit_shift = (uint8_t)bs_r_bits(r, 3);
    bs_r_align(r);
}

VL264_INTERNAL void write_pps(bs_writer* w) {
    // Minimal PPS — nothing beyond what SPS covers for our use case.
    bs_w_ue(w, 0); // pps_id
    bs_w_align(w);
}

VL264_INTERNAL void read_pps(bs_reader* r) {
    bs_r_ue(r); // pps_id
    bs_r_align(r);
}

VL264_INTERNAL void write_slice_hdr(bs_writer* w, const vl264_slice_hdr* h) {
    bs_w_ue(w, h->slice_idx);
    bs_w_bit1(w, h->is_idr);
    bs_w_se(w, h->qp_delta);
    if (!h->is_idr) {
        bs_w_ue(w, h->ref_idx);
    }
    bs_w_align(w);
}

VL264_INTERNAL void read_slice_hdr(bs_reader* r, vl264_slice_hdr* h) {
    h->slice_idx = bs_r_ue(r);
    h->is_idr = (uint8_t)bs_r_bit1(r);
    h->qp_delta = (int8_t)bs_r_se(r);
    if (!h->is_idr) {
        h->ref_idx = bs_r_ue(r);
    } else {
        h->ref_idx = 0;
    }
    bs_r_align(r);
}

// ═════════════════════════════════════════════════════════════════════════════
// Section 5: 4x4 integer DCT (H.264 spec)
// ═════════════════════════════════════════════════════════════════════════════

// Forward 4x4 integer DCT (H.264 section 8.5.10)
// Core matrix Cf:
//   [ 1  1  1  1 ]
//   [ 2  1 -1 -2 ]
//   [ 1 -1 -1  1 ]
//   [ 1 -2  2 -1 ]
VL264_INTERNAL VL264_HOT void dct4x4_fwd(const int16_t in[static restrict 16], int16_t out[static restrict 16]) {
    int16_t tmp[16];

    // Horizontal pass
    for (int i = 0; i < 4; i++) {
        int s0 = in[i*4+0], s1 = in[i*4+1], s2 = in[i*4+2], s3 = in[i*4+3];
        int d0 = s0 + s3, d1 = s1 + s2, d2 = s1 - s2, d3 = s0 - s3;
        tmp[i*4+0] = (int16_t)(d0 + d1);
        tmp[i*4+1] = (int16_t)(2 * d3 + d2);
        tmp[i*4+2] = (int16_t)(d0 - d1);
        tmp[i*4+3] = (int16_t)(d3 - 2 * d2);
    }

    // Vertical pass
    for (int j = 0; j < 4; j++) {
        int s0 = tmp[0*4+j], s1 = tmp[1*4+j], s2 = tmp[2*4+j], s3 = tmp[3*4+j];
        int d0 = s0 + s3, d1 = s1 + s2, d2 = s1 - s2, d3 = s0 - s3;
        out[0*4+j] = (int16_t)(d0 + d1);
        out[1*4+j] = (int16_t)(2 * d3 + d2);
        out[2*4+j] = (int16_t)(d0 - d1);
        out[3*4+j] = (int16_t)(d3 - 2 * d2);
    }
}

// Inverse 4x4 integer DCT
VL264_INTERNAL VL264_HOT void dct4x4_inv(const int16_t in[static restrict 16], int16_t out[static restrict 16]) {
    int16_t tmp[16];

    // Horizontal pass
    for (int i = 0; i < 4; i++) {
        int s0 = in[i*4+0], s1 = in[i*4+1], s2 = in[i*4+2], s3 = in[i*4+3];
        int d0 = s0 + s2;
        int d1 = s0 - s2;
        int d2 = (s1 >> 1) - s3;
        int d3 = s1 + (s3 >> 1);
        tmp[i*4+0] = (int16_t)(d0 + d3);
        tmp[i*4+1] = (int16_t)(d1 + d2);
        tmp[i*4+2] = (int16_t)(d1 - d2);
        tmp[i*4+3] = (int16_t)(d0 - d3);
    }

    // Vertical pass + round + shift
    for (int j = 0; j < 4; j++) {
        int s0 = tmp[0*4+j], s1 = tmp[1*4+j], s2 = tmp[2*4+j], s3 = tmp[3*4+j];
        int d0 = s0 + s2;
        int d1 = s0 - s2;
        int d2 = (s1 >> 1) - s3;
        int d3 = s1 + (s3 >> 1);
        out[0*4+j] = (int16_t)((d0 + d3 + 8) >> 4);
        out[1*4+j] = (int16_t)((d1 + d2 + 8) >> 4);
        out[2*4+j] = (int16_t)((d1 - d2 + 8) >> 4);
        out[3*4+j] = (int16_t)((d0 - d3 + 8) >> 4);
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// Section 6: Quantization
// ═════════════════════════════════════════════════════════════════════════════

// H.264 quantization MF (multiplication factor) for QP%6
// Positions: [qp%6][{0,0}, {0,2}, {2,2}]
static const uint16_t quant_mf[6][3] = {
    {13107, 5243, 8066},
    {11916, 4660, 7490},
    {10082, 4194, 6554},
    { 9362, 3647, 5825},
    { 8192, 3355, 5243},
    { 7282, 2893, 4559},
};

// Dequant scale factors
static const uint16_t dequant_v[6][3] = {
    {10, 16, 13},
    {11, 18, 14},
    {13, 20, 16},
    {14, 23, 18},
    {16, 25, 20},
    {18, 29, 23},
};

// Position-to-MF-index map for a 4x4 block
static const uint8_t mf_idx[16] = {
    0,2,0,2,
    2,1,2,1,
    0,2,0,2,
    2,1,2,1,
};

// Zigzag scan order for 4x4 block
static const uint8_t zigzag4x4[16] = {
    0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15
};

VL264_INTERNAL VL264_HOT void quant4x4(int16_t coeff[16], int32_t qp, bool is_intra) {
    int32_t qp_rem = qp % 6;
    int32_t qp_div = qp / 6;
    // Wider dead zone for inter blocks: /4 instead of /6 produces more zeros
    // (standard H.264 uses /3 intra, /6 inter; we use /3 intra, /4 inter)
    int32_t f = is_intra ? (1 << (15 + qp_div)) / 3 : (1 << (15 + qp_div)) / 4;

    for (int i = 0; i < 16; i++) {
        int32_t mf = quant_mf[qp_rem][mf_idx[i]];
        int32_t sign = coeff[i] < 0 ? -1 : 1;
        int32_t absval = abs(coeff[i]);
        coeff[i] = (int16_t)(sign * ((absval * mf + f) >> (15 + qp_div)));
    }
}

VL264_INTERNAL VL264_HOT void dequant4x4(int16_t coeff[16], int32_t qp) {
    int32_t qp_rem = qp % 6;
    int32_t qp_div = qp / 6;

    for (int i = 0; i < 16; i++) {
        int32_t v = dequant_v[qp_rem][mf_idx[i]];
        int32_t scaled = (int32_t)coeff[i] * v;
        if (qp_div >= 2) {
            coeff[i] = (int16_t)(scaled * (1 << (qp_div - 2)));
        } else {
            coeff[i] = (int16_t)((scaled + (1 << (1 - qp_div))) >> (2 - qp_div));
        }
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// Section 7: Intra prediction
// ═════════════════════════════════════════════════════════════════════════════

// Get neighbor pixels for a 4x4 block at position (bx,by) in a 128-wide slice.
// top[0..3]: pixels above, left[0..3]: pixels to the left, tl: top-left corner.
// Returns true if all neighbors available.
VL264_INTERNAL bool get_neighbors(const int16_t* recon, int32_t bx, int32_t by,
                                   int16_t top[4], int16_t left[4], int16_t* tl) {
    bool has_top = by > 0;
    bool has_left = bx > 0;

    if (has_top) {
        int32_t y = by * 4 - 1;
        for (int i = 0; i < 4; i++)
            top[i] = recon[y * DIM + bx * 4 + i];
    } else {
        for (int i = 0; i < 4; i++) top[i] = 128;
    }

    if (has_left) {
        int32_t x = bx * 4 - 1;
        for (int i = 0; i < 4; i++)
            left[i] = recon[(by * 4 + i) * DIM + x];
    } else {
        for (int i = 0; i < 4; i++) left[i] = 128;
    }

    if (has_top && has_left) {
        *tl = recon[(by * 4 - 1) * DIM + bx * 4 - 1];
    } else if (has_top) {
        *tl = top[0];
    } else if (has_left) {
        *tl = left[0];
    } else {
        *tl = 128;
    }

    return has_top || has_left;
}

VL264_INTERNAL void intra_pred_4x4(int16_t pred[static 16], int32_t mode,
                                     const int16_t top[4], const int16_t left[4],
                                     int16_t tl) {
    switch (mode) {
    case INTRA_VERT:
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                pred[y*4+x] = top[x];
        break;

    case INTRA_HORIZ:
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                pred[y*4+x] = left[y];
        break;

    case INTRA_DC: {
        int32_t sum = 0;
        for (int i = 0; i < 4; i++) sum += top[i] + left[i];
        int16_t dc = (int16_t)((sum + 4) >> 3);
        for (int i = 0; i < 16; i++) pred[i] = dc;
        break;
    }

    case INTRA_DDL: // diagonal down-left
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++) {
                int idx = x + y;
                if (idx < 3)
                    pred[y*4+x] = (int16_t)((top[idx] + 2*top[idx+1] + top[VL264_MIN(idx+2, 3)] + 2) >> 2);
                else
                    pred[y*4+x] = (int16_t)((top[3] + 2*top[3] + top[3] + 2) >> 2);
            }
        break;

    case INTRA_DDR: // diagonal down-right
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++) {
                int d = x - y;
                if (d > 0)
                    pred[y*4+x] = (int16_t)((top[d-1] + 2*top[d] + top[VL264_MIN(d+1, 3)] + 2) >> 2);
                else if (d == 0)
                    pred[y*4+x] = (int16_t)((left[0] + 2*tl + top[0] + 2) >> 2);
                else
                    pred[y*4+x] = (int16_t)((left[-d-1] + 2*left[-d] + ((-d+1 < 4) ? left[-d+1] : left[3]) + 2) >> 2);
            }
        break;

    case INTRA_VR: // vertical-right
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++) {
                int zVR = 2*x - y;
                if (zVR >= 0 && (zVR & 1) == 0) {
                    int idx = zVR >> 1;
                    if (idx == 0) pred[y*4+x] = (int16_t)((tl + top[0] + 1) >> 1);
                    else pred[y*4+x] = (int16_t)((top[idx-1] + top[idx] + 1) >> 1);
                } else if (zVR >= 0) {
                    int idx = zVR >> 1;
                    if (idx == 0) pred[y*4+x] = (int16_t)((left[0] + 2*tl + top[0] + 2) >> 2);
                    else pred[y*4+x] = (int16_t)((top[idx-1] + 2*top[idx] + top[VL264_MIN(idx+1,3)] + 2) >> 2);
                } else if (zVR == -1) {
                    pred[y*4+x] = (int16_t)((tl + 2*left[0] + left[1] + 2) >> 2);
                } else {
                    int idx = (-zVR) / 2;
                    pred[y*4+x] = (int16_t)((left[idx-1] + 2*left[idx] + left[VL264_MIN(idx+1,3)] + 2) >> 2);
                }
            }
        break;

    case INTRA_HD: // horizontal-down
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++) {
                int zHD = 2*y - x;
                if (zHD >= 0 && (zHD & 1) == 0) {
                    int idx = zHD >> 1;
                    if (idx == 0) pred[y*4+x] = (int16_t)((tl + left[0] + 1) >> 1);
                    else pred[y*4+x] = (int16_t)((left[idx-1] + left[idx] + 1) >> 1);
                } else if (zHD >= 0) {
                    int idx = zHD >> 1;
                    if (idx == 0) pred[y*4+x] = (int16_t)((top[0] + 2*tl + left[0] + 2) >> 2);
                    else pred[y*4+x] = (int16_t)((left[idx-1] + 2*left[idx] + left[VL264_MIN(idx+1,3)] + 2) >> 2);
                } else if (zHD == -1) {
                    pred[y*4+x] = (int16_t)((tl + 2*top[0] + top[1] + 2) >> 2);
                } else {
                    int idx = (-zHD) / 2;
                    pred[y*4+x] = (int16_t)((top[idx-1] + 2*top[idx] + top[VL264_MIN(idx+1,3)] + 2) >> 2);
                }
            }
        break;

    case INTRA_VL: // vertical-left
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++) {
                int idx = x + (y >> 1);
                if ((y & 1) == 0)
                    pred[y*4+x] = (int16_t)((top[VL264_MIN(idx,3)] + top[VL264_MIN(idx+1,3)] + 1) >> 1);
                else
                    pred[y*4+x] = (int16_t)((top[VL264_MIN(idx,3)] + 2*top[VL264_MIN(idx+1,3)] + top[VL264_MIN(idx+2,3)] + 2) >> 2);
            }
        break;

    case INTRA_HU: // horizontal-up
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++) {
                int zHU = x + 2*y;
                if (zHU < 6 && (zHU & 1) == 0) {
                    int idx = zHU >> 1;
                    pred[y*4+x] = (int16_t)((left[idx] + left[idx+1] + 1) >> 1);
                } else if (zHU < 6) {
                    int idx = zHU >> 1;
                    pred[y*4+x] = (int16_t)((left[idx] + 2*left[idx+1] + left[VL264_MIN(idx+2,3)] + 2) >> 2);
                } else if (zHU == 6) {
                    pred[y*4+x] = (int16_t)((left[3] + 3*left[3] + 2) >> 2);
                } else {
                    pred[y*4+x] = left[3];
                }
            }
        break;

    default:
        // Fallback to DC
        for (int i = 0; i < 16; i++) pred[i] = 128;
        break;
    }
}

// SAD for 4x4 — branchless abs via arithmetic
VL264_INTERNAL VL264_HOT VL264_PURE int32_t sad_4x4(const int16_t a[static 16], const int16_t b[static 16]) {
    int32_t sum = 0;
    for (int i = 0; i < 16; i++) {
        int32_t d = (int32_t)a[i] - (int32_t)b[i];
        sum += (d ^ (d >> 31)) - (d >> 31); // branchless abs
    }
    return sum;
}

// Intra mode decision with restricted fast-path
// Returns best mode. Writes predicted block to pred_out.


// ═════════════════════════════════════════════════════════════════════════════
// Section 8: Inter prediction / Motion estimation
// ═════════════════════════════════════════════════════════════════════════════

typedef struct { int16_t x, y; } vl264_mv;

// Get a 4x4 block from a slice buffer. Fast path for interior blocks.
VL264_INTERNAL VL264_HOT void get_block(const int16_t* slice, int32_t px, int32_t py,
                               int16_t out[static 16]) {
    // Fast path: block is fully inside the slice (most common)
    if (VL264_LIKELY(px >= 0 && px + 3 < DIM && py >= 0 && py + 3 < DIM)) {
        const int16_t* src = slice + py * DIM + px;
        for (int dy = 0; dy < 4; dy++) {
            out[dy*4+0] = src[0]; out[dy*4+1] = src[1];
            out[dy*4+2] = src[2]; out[dy*4+3] = src[3];
            src += DIM;
        }
        return;
    }
    // Slow path: clamp at boundaries
    for (int dy = 0; dy < 4; dy++) {
        for (int dx = 0; dx < 4; dx++) {
            int32_t x = VL264_CLAMP(px + dx, 0, DIM - 1);
            int32_t y = VL264_CLAMP(py + dy, 0, DIM - 1);
            out[dy*4+dx] = slice[y * DIM + x];
        }
    }
}

// Diamond search with early termination. Much faster than exhaustive ±3.
// Pattern: check center → 4 diamond → 4 diagonal → refine best direction
VL264_INTERNAL vl264_mv me_search_int(const int16_t orig[static 16], const int16_t* ref,
                                       int32_t bx, int32_t by, int32_t range,
                                       int32_t* best_sad) {
    int32_t px = bx * 4, py = by * 4;
    vl264_mv best = {0, 0};
    *best_sad = INT32_MAX;
    int16_t blk[16];

    // Check center (zero MV) first
    get_block(ref, px, py, blk);
    *best_sad = sad_4x4(orig, blk);
    if (*best_sad == 0) return best; // perfect match

    // Diamond search: expanding rings
    static const int8_t diamond[][2] = {
        {-1,0},{1,0},{0,-1},{0,1},  // cross
        {-1,-1},{1,-1},{-1,1},{1,1} // diagonal
    };

    for (int step = 1; step <= range; step++) {
        bool improved = false;
        for (int d = 0; d < 8; d++) {
            int32_t dx = diamond[d][0] * step;
            int32_t dy = diamond[d][1] * step;
            get_block(ref, px + dx, py + dy, blk);
            int32_t cost = sad_4x4(orig, blk);
            if (cost < *best_sad) {
                *best_sad = cost;
                best.x = (int16_t)(dx * 4);
                best.y = (int16_t)(dy * 4);
                improved = true;
                if (cost == 0) return best; // perfect
            }
        }
        if (!improved) break; // no improvement at this radius, stop expanding
    }
    return best;
}

// Bilinear interpolation at half/quarter-pel position
VL264_INTERNAL int16_t interp_pixel(const int16_t* ref, int32_t qx, int32_t qy) {
    int32_t ix = qx >> 2;
    int32_t iy = qy >> 2;
    int32_t fx = qx & 3;
    int32_t fy = qy & 3;

    int32_t x0 = VL264_CLAMP(ix, 0, DIM - 1);
    int32_t x1 = VL264_CLAMP(ix + 1, 0, DIM - 1);
    int32_t y0 = VL264_CLAMP(iy, 0, DIM - 1);
    int32_t y1 = VL264_CLAMP(iy + 1, 0, DIM - 1);

    int32_t v00 = ref[y0 * DIM + x0];
    int32_t v10 = ref[y0 * DIM + x1];
    int32_t v01 = ref[y1 * DIM + x0];
    int32_t v11 = ref[y1 * DIM + x1];

    int32_t val = (4 - fx) * (4 - fy) * v00 +
                  fx * (4 - fy) * v10 +
                  (4 - fx) * fy * v01 +
                  fx * fy * v11;
    return (int16_t)((val + 8) >> 4);
}

// Get a 4x4 block at sub-pixel position
VL264_INTERNAL void mc_block(const int16_t* ref, int32_t px, int32_t py,
                              vl264_mv mv, int16_t out[static 16]) {
    int32_t qx = px * 4 + mv.x;
    int32_t qy = py * 4 + mv.y;
    for (int dy = 0; dy < 4; dy++) {
        for (int dx = 0; dx < 4; dx++) {
            out[dy*4+dx] = interp_pixel(ref, qx + dx * 4, qy + dy * 4);
        }
    }
}

// Quarter-pel refinement around integer result
VL264_INTERNAL vl264_mv me_refine_qpel(const int16_t orig[16], const int16_t* ref,
                                         int32_t bx, int32_t by, vl264_mv start,
                                         int32_t* best_sad) {
    int32_t px = bx * 4, py = by * 4;
    vl264_mv best = start;
    int16_t blk[16];

    // Diamond pattern at +/-1 and +/-2 quarter-pel
    static const int8_t offsets[][2] = {
        {-1,0},{1,0},{0,-1},{0,1},
        {-2,0},{2,0},{0,-2},{0,2},
        {-1,-1},{1,-1},{-1,1},{1,1},
    };

    for (int i = 0; i < 12; i++) {
        vl264_mv trial = {
            (int16_t)(start.x + offsets[i][0]),
            (int16_t)(start.y + offsets[i][1])
        };
        mc_block(ref, px, py, trial, blk);
        int32_t cost = sad_4x4(orig, blk);
        if (cost < *best_sad) {
            *best_sad = cost;
            best = trial;
        }
    }
    return best;
}

// ═════════════════════════════════════════════════════════════════════════════
// Section 9 & 10: CAVLC entropy coding + lookup tables
// ═════════════════════════════════════════════════════════════════════════════

// Rice (Golomb-Rice) coding for signed coefficient levels.
// Unary prefix for quotient, k-bit suffix for remainder, sign bit.
// Escape: if quotient > 12, write 12 zeros + stop + ue(quotient-12) instead.
VL264_INTERNAL VL264_HOT void rice_encode(bs_writer* w, int32_t val, int32_t k) {
    uint32_t absval = (uint32_t)abs(val);
    uint32_t q = absval >> k;
    uint32_t rem = absval & (shl32(1u, k) - 1);
    if (q < 12) {
        // Normal: q zeros + stop bit
        for (uint32_t i = 0; i < q; i++) bs_w_bit1(w, 0);
        bs_w_bit1(w, 1);
    } else {
        // Escape: 12 zeros (no stop) + ue(quotient)
        for (int i = 0; i < 12; i++) bs_w_bit1(w, 0);
        bs_w_ue(w, q);
    }
    if (k > 0) bs_w_bits(w, rem, k);
    if (absval > 0) bs_w_bit1(w, val < 0 ? 1 : 0);
}

VL264_INTERNAL VL264_HOT int32_t rice_decode(bs_reader* r, int32_t k) {
    uint32_t q = 0;
    for (;;) {
        if (bs_r_bit1(r)) break; // stop bit → normal path (q < 12)
        q++;
        if (q >= 12) {
            q = bs_r_ue(r); // escape: read full quotient via ue
            break;
        }
    }
    uint32_t rem = (k > 0) ? bs_r_bits(r, k) : 0;
    uint32_t absval = (q << k) + rem;
    if (absval == 0) return 0;
    int32_t sign = bs_r_bit1(r) ? -1 : 1;
    return sign * (int32_t)absval;
}

// Compact coefficient coding.
// 3 modes signaled by 2-bit header:
//   00 = all zero (1 bit total with the leading 0)
//   01 = DC-only: just se(dc_level) — most common case at high QP
//   10 = sparse: ue(total_coeff-2) + ue(last_sig) + sig_map + levels
//   11 = dense:  16 × se(level) — for low QP when most coefficients are non-zero

VL264_INTERNAL VL264_HOT void cavlc_encode_block(bs_writer* w, const int16_t coeff[static 16], int32_t nc) {
    (void)nc;

    int32_t total_coeff = 0;
    int32_t last_sig = -1;
    int16_t levels[16];
    uint8_t sig_map[16] = {0};

    for (int i = 0; i < 16; i++) {
        int16_t v = coeff[zigzag4x4[i]];
        if (v != 0) {
            sig_map[i] = 1;
            levels[total_coeff++] = v;
            last_sig = i;
        }
    }

    if (total_coeff == 0) {
        bs_w_bits(w, 0, 2); // 00
        return;
    }

    if (total_coeff == 1) {
        bs_w_bits(w, 1, 2); // 01
        bs_w_ue(w, (uint32_t)last_sig);
        rice_encode(w, levels[0], 0);
        return;
    }

    if (total_coeff >= 12) {
        bs_w_bits(w, 3, 2); // 11
        for (int i = 0; i < 16; i++)
            rice_encode(w, coeff[zigzag4x4[i]], 1); // k=1 for dense
        return;
    }

    // Sparse (2-11 coefficients)
    bs_w_bits(w, 2, 2); // 10
    bs_w_ue(w, (uint32_t)(total_coeff - 2));
    bs_w_ue(w, (uint32_t)last_sig);

    for (int i = 0; i < last_sig; i++)
        bs_w_bit1(w, sig_map[i]);

    for (int i = total_coeff - 1; i >= 0; i--)
        rice_encode(w, levels[i], 0); // k=0 for sparse
}

VL264_INTERNAL VL264_HOT void cavlc_decode_block(bs_reader* r, int16_t coeff[static 16], int32_t nc) {
    (void)nc;
    memset(coeff, 0, 16 * sizeof(int16_t));

    uint32_t hdr = bs_r_bits(r, 2);

    if (hdr == 0) return; // all zero

    if (hdr == 1) {
        uint32_t pos = bs_r_ue(r);
        if (pos > 15) pos = 15;
        coeff[zigzag4x4[pos]] = (int16_t)rice_decode(r, 0);
        return;
    }

    if (hdr == 3) {
        for (int i = 0; i < 16; i++)
            coeff[zigzag4x4[i]] = (int16_t)rice_decode(r, 1); // k=1 for dense
        return;
    }

    // Sparse (hdr == 2)
    uint32_t total_coeff = bs_r_ue(r) + 2;
    if (total_coeff > 16) total_coeff = 16;

    uint32_t last_sig = bs_r_ue(r);
    if (last_sig > 15) last_sig = 15;

    uint8_t sig_map[16] = {0};
    sig_map[last_sig] = 1;
    for (uint32_t i = 0; i < last_sig; i++)
        sig_map[i] = (uint8_t)bs_r_bit1(r);

    int16_t levels[16];
    for (int i = (int)total_coeff - 1; i >= 0; i--)
        levels[i] = (int16_t)rice_decode(r, 0); // k=0 for sparse

    int32_t li = 0;
    for (uint32_t i = 0; i <= last_sig && li < (int32_t)total_coeff; i++) {
        if (sig_map[i])
            coeff[zigzag4x4[i]] = levels[li++];
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Section 11: Axis selection
// ═════════════════════════════════════════════════════════════════════════════

// Compute sum of squared differences between adjacent slices along an axis
// Subsampled axis cost: check every STEP-th voxel, every SSTEP-th slice pair.
// 128/4=32 samples per dimension, 128/8=16 slice pairs -> 32*32*16 = 16K ops per axis.
#define AXIS_STEP  4
#define AXIS_SSTEP 8

VL264_INTERNAL double axis_cost(const uint8_t* data, int32_t axis) {
    double total = 0.0;
    for (int32_t s = 0; s < DIM - 1; s += AXIS_SSTEP) {
        for (int32_t a = 0; a < DIM; a += AXIS_STEP) {
            for (int32_t b = 0; b < DIM; b += AXIS_STEP) {
                int32_t i0, i1;
                switch (axis) {
                case 0:
                    i0 = a * DIM * DIM + b * DIM + s;
                    i1 = a * DIM * DIM + b * DIM + s + 1;
                    break;
                case 1:
                    i0 = a * DIM * DIM + s * DIM + b;
                    i1 = a * DIM * DIM + (s + 1) * DIM + b;
                    break;
                default:
                    i0 = s * DIM * DIM + a * DIM + b;
                    i1 = (s + 1) * DIM * DIM + a * DIM + b;
                    break;
                }
                int32_t d = (int32_t)data[i0] - (int32_t)data[i1];
                total += (double)(d * d);
            }
        }
    }
    return total;
}

VL264_INTERNAL int32_t select_best_axis(const uint8_t* data) {
    double cx = axis_cost(data, 0);
    double cy = axis_cost(data, 1);
    double cz = axis_cost(data, 2);
    if (cx <= cy && cx <= cz) return 0;
    if (cy <= cz) return 1;
    return 2;
}

// ═════════════════════════════════════════════════════════════════════════════
// Section 12: Slice extract / insert
// ═════════════════════════════════════════════════════════════════════════════

// Extract a 128x128 slice from a 128^3 volume along the given axis.
// Output is int16_t (widened from u8).
VL264_INTERNAL void extract_slice(const uint8_t* VL264_RESTRICT vol,
                                   int32_t axis, int32_t idx,
                                   int16_t* VL264_RESTRICT out) {
    if (axis == 2) {
        // Z-axis: rows are contiguous in memory — fast path
        const uint8_t* src = vol + idx * DIM * DIM;
        for (int32_t i = 0; i < DIM * DIM; i++)
            out[i] = (int16_t)src[i];
        return;
    }
    for (int32_t a = 0; a < DIM; a++) {
        for (int32_t b = 0; b < DIM; b++) {
            int32_t vi;
            switch (axis) {
            case 0: vi = a * DIM * DIM + b * DIM + idx; break;
            default: vi = a * DIM * DIM + idx * DIM + b; break;
            }
            out[a * DIM + b] = (int16_t)vol[vi];
        }
    }
}

// Insert a decoded int16_t slice back into a u8 volume (with clamping).
VL264_INTERNAL void insert_slice(uint8_t* VL264_RESTRICT vol,
                                  int32_t axis, int32_t idx,
                                  const int16_t* VL264_RESTRICT slice) {
    for (int32_t a = 0; a < DIM; a++) {
        for (int32_t b = 0; b < DIM; b++) {
            int32_t vi;
            switch (axis) {
            case 0: vi = a * DIM * DIM + b * DIM + idx; break;
            case 1: vi = a * DIM * DIM + idx * DIM + b; break;
            default: vi = idx * DIM * DIM + a * DIM + b; break;
            }
            int32_t v = slice[a * DIM + b];
            vol[vi] = (uint8_t)VL264_CLAMP(v, 0, 255);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Section 13: Morton (Z-curve) ordering
// ═════════════════════════════════════════════════════════════════════════════

// Bit-reverse / interleave for 7-bit indices (128 slices -> Morton order)
// For 1D Morton on 128 elements, we use bit-reversal of the 7-bit index
// to create a traversal that alternates between coarse and fine scales.
VL264_INTERNAL void morton_order_128(uint32_t order[128]) {
    // Simple bit-reversal ordering for 7-bit indices
    for (uint32_t i = 0; i < 128; i++) {
        uint32_t rev = 0;
        uint32_t v = i;
        for (int b = 0; b < 7; b++) {
            rev = (rev << 1) | (v & 1);
            v >>= 1;
        }
        order[i] = rev;
    }
}

VL264_INTERNAL void raster_order_128(uint32_t order[128]) {
    for (uint32_t i = 0; i < 128; i++) order[i] = i;
}

// Should this slice be forced to I-frame?
VL264_INTERNAL bool should_force_iframe(const uint32_t* order, uint32_t coded_idx,
                                         int32_t interval) {
    if (coded_idx == 0) return true;
    if (interval > 0 && (int32_t)(coded_idx % (uint32_t)interval) == 0) return true;
    int32_t cur = (int32_t)order[coded_idx];
    int32_t prev = (int32_t)order[coded_idx - 1];
    return abs(cur - prev) > 8;
}

// ═════════════════════════════════════════════════════════════════════════════
// Section 14: Boundary prediction
// ═════════════════════════════════════════════════════════════════════════════

// Get the boundary face from neighbors for a given axis and slice index.
// Returns NULL if no boundary face available.
VL264_INTERNAL const uint8_t* get_boundary_face(const vl264_neighbors* n,
                                                  int32_t axis, int32_t slice_idx) {
    if (!n) return NULL;
    // First slice along axis -> negative face
    if (slice_idx == 0) {
        switch (axis) {
        case 0: return n->neg_x;
        case 1: return n->neg_y;
        case 2: return n->neg_z;
        }
    }
    // Last slice along axis -> positive face
    if (slice_idx == DIM - 1) {
        switch (axis) {
        case 0: return n->pos_x;
        case 1: return n->pos_y;
        case 2: return n->pos_z;
        }
    }
    return NULL;
}

// Widen a u8 boundary face to int16_t for use as reference
VL264_INTERNAL void widen_face(const uint8_t* VL264_RESTRICT face,
                                int16_t* VL264_RESTRICT out) {
    for (int32_t i = 0; i < DIM * DIM; i++) {
        out[i] = (int16_t)face[i];
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Section 15: LOD delta coding
// ═════════════════════════════════════════════════════════════════════════════

// Upsample a (dim)^3 chunk to 128^3 using bilinear interpolation.
// Output is int16_t.
VL264_INTERNAL void upsample_2x(const uint8_t* VL264_RESTRICT coarse, int32_t cdim,
                                  int16_t* VL264_RESTRICT fine) {
    float scale = (float)cdim / (float)DIM;
    for (int32_t z = 0; z < DIM; z++) {
        for (int32_t y = 0; y < DIM; y++) {
            for (int32_t x = 0; x < DIM; x++) {
                float sx = ((float)x + 0.5f) * scale - 0.5f;
                float sy = ((float)y + 0.5f) * scale - 0.5f;
                float sz = ((float)z + 0.5f) * scale - 0.5f;

                int32_t x0 = VL264_CLAMP((int32_t)sx, 0, cdim - 1);
                int32_t y0 = VL264_CLAMP((int32_t)sy, 0, cdim - 1);
                int32_t z0 = VL264_CLAMP((int32_t)sz, 0, cdim - 1);
                int32_t x1 = VL264_MIN(x0 + 1, cdim - 1);
                int32_t y1 = VL264_MIN(y0 + 1, cdim - 1);
                int32_t z1 = VL264_MIN(z0 + 1, cdim - 1);

                float fx = sx - (float)x0;
                float fy = sy - (float)y0;
                float fz = sz - (float)z0;
                fx = VL264_CLAMP(fx, 0.0f, 1.0f);
                fy = VL264_CLAMP(fy, 0.0f, 1.0f);
                fz = VL264_CLAMP(fz, 0.0f, 1.0f);

                #define C(xx,yy,zz) ((float)coarse[(zz)*cdim*cdim + (yy)*cdim + (xx)])
                float v = C(x0,y0,z0)*(1-fx)*(1-fy)*(1-fz)
                        + C(x1,y0,z0)*fx*(1-fy)*(1-fz)
                        + C(x0,y1,z0)*(1-fx)*fy*(1-fz)
                        + C(x1,y1,z0)*fx*fy*(1-fz)
                        + C(x0,y0,z1)*(1-fx)*(1-fy)*fz
                        + C(x1,y0,z1)*fx*(1-fy)*fz
                        + C(x0,y1,z1)*(1-fx)*fy*fz
                        + C(x1,y1,z1)*fx*fy*fz;
                #undef C
                fine[z * DIM * DIM + y * DIM + x] = (int16_t)(v + 0.5f);
            }
        }
    }
}

// Extract a single 128x128 slice from the upsampled LOD volume
VL264_INTERNAL void lod_extract_slice(const int16_t* upsampled,
                                       int32_t axis, int32_t idx,
                                       int16_t* out) {
    for (int32_t a = 0; a < DIM; a++) {
        for (int32_t b = 0; b < DIM; b++) {
            int32_t vi;
            switch (axis) {
            case 0: vi = a * DIM * DIM + b * DIM + idx; break;
            case 1: vi = a * DIM * DIM + idx * DIM + b; break;
            default: vi = idx * DIM * DIM + a * DIM + b; break;
            }
            out[a * DIM + b] = upsampled[vi];
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// ═════════════════════════════════════════════════════════════════════════════
// Section 16: Helpers
// ═════════════════════════════════════════════════════════════════════════════

VL264_INTERNAL void clamp_i16_u8(uint8_t* VL264_RESTRICT dst,
                                  const int16_t* VL264_RESTRICT src, int32_t n) {
    for (int32_t i = 0; i < n; i++) {
        dst[i] = (uint8_t)VL264_CLAMP(src[i], 0, 255);
    }
}

// Detect effective bit depth by finding GCD of a sample of non-zero values.
// Returns the right-shift amount (0-5). 0 = full 8-bit, 3 = 5 significant bits, etc.
VL264_INTERNAL int32_t detect_bit_shift(const uint8_t* data) {
    // Check if ALL values have their low N bits zeroed.
    // This is stricter than GCD — ensures lossless round-trip for the shift.
    uint8_t low_bits_or = 0;
    for (size_t i = 0; i < VL264_CHUNK_VOXELS; i += 64)
        low_bits_or |= data[i];
    // Find how many trailing zero bits are common to all values
    if (low_bits_or & 1) return 0;   // at least one odd value
    if (low_bits_or & 2) return 1;   // bit 1 is set somewhere
    if (low_bits_or & 4) return 2;   // bit 2 is set somewhere
    // Check full volume (not just sample) for shift >= 3
    low_bits_or = 0;
    for (size_t i = 0; i < VL264_CHUNK_VOXELS; i++)
        low_bits_or |= data[i];
    if ((low_bits_or & 0x07) == 0) return 3; // low 3 bits all zero everywhere
    if ((low_bits_or & 0x03) == 0) return 2;
    if ((low_bits_or & 0x01) == 0) return 1;
    return 0;
}

// 3x3 box filter for reference frame smoothing.
// Reduces noise in the prediction reference, making residuals smaller.
// ═════════════════════════════════════════════════════════════════════════════
// Section 18: Encoder state + core
// ═════════════════════════════════════════════════════════════════════════════

struct vl264_enc {
    vl264_cfg        cfg;
    vl264_enc_stats  stats;
    int32_t          resolved_qp;

    VL264_ALIGNED(64) int16_t cur_slice[DIM * DIM];
    VL264_ALIGNED(64) int16_t ref_slice[DIM * DIM];
    VL264_ALIGNED(64) int16_t ref_smooth[DIM * DIM]; // smoothed reference for inter pred
    VL264_ALIGNED(64) int16_t recon_slice[DIM * DIM];

    int16_t* lod_upsampled;

    VL264_ALIGNED(64) int16_t boundary_buf[DIM * DIM];

    int16_t nc_map[BSTRIDE * BSTRIDE];
    int16_t dc_map[BSTRIDE * BSTRIDE];
    vl264_mv mv_map[BSTRIDE * BSTRIDE]; // MVs for MV prediction

    uint32_t slice_order[DIM];
};

// Block encode state — holds results of prediction + transform + quant
typedef struct {
    int16_t pred[16];
    int16_t orig[16]; // original block data (needed for skip fallback)
    int16_t coeff[16]; // quantized
    int32_t total_coeff;
    int32_t mb_type;
    int32_t mode;
    vl264_mv mv;
} block_state_t;

// Phase 1: Compute prediction, residual, DCT, quant. Write reconstruction.
// Returns total_coeff (0 = skip candidate).
VL264_INTERNAL VL264_HOT VL264_FLATTEN int32_t try_encode_block(vl264_enc* e, block_state_t* bs,
                                         const int16_t* cur_slice, int16_t* recon_slice,
                                         const int16_t* ref_slice,
                                         int32_t bx, int32_t by,
                                         bool is_iframe, int32_t qp) {
    int16_t* orig = bs->orig;
    int16_t zcoeff[16] = {0}; // zero-MV quantized coefficients (reused later)
    for (int dy = 0; dy < 4; dy++)
        for (int dx = 0; dx < 4; dx++)
            orig[dy*4+dx] = cur_slice[(by*4+dy) * DIM + bx*4+dx];

    bs->mv = (vl264_mv){0, 0};
    bs->mode = 0;

    if (is_iframe || !ref_slice) {
        int16_t top[4], left[4], tl;
        get_neighbors(recon_slice, bx, by, top, left, &tl);
        // FAST: DC only. DEFAULT: DC+H+V. MAX: all modes.
        intra_pred_4x4(bs->pred, INTRA_DC, top, left, tl);
        bs->mode = INTRA_DC;
        if (e->cfg.quality >= VL264_DEFAULT) {
            int32_t best = sad_4x4(orig, bs->pred);
            int16_t tmp[16];
            intra_pred_4x4(tmp, INTRA_HORIZ, top, left, tl);
            int32_t c = sad_4x4(orig, tmp);
            if (c < best) { best = c; bs->mode = INTRA_HORIZ; memcpy(bs->pred, tmp, 32); }
            intra_pred_4x4(tmp, INTRA_VERT, top, left, tl);
            c = sad_4x4(orig, tmp);
            if (c < best) { best = c; bs->mode = INTRA_VERT; memcpy(bs->pred, tmp, 32); }
            if (e->cfg.quality >= VL264_MAX) {
                for (int m = INTRA_DDL; m < INTRA_NUM_MODES; m++) {
                    intra_pred_4x4(tmp, m, top, left, tl);
                    c = sad_4x4(orig, tmp);
                    if (c < best) { best = c; bs->mode = m; memcpy(bs->pred, tmp, 32); }
                }
            }
        }
        bs->mb_type = MB_TYPE_I4x4;
    } else {
        // P-frame: zero-MV first. Only search if zero-MV produces many non-zero coefficients.
        get_block(ref_slice, bx*4, by*4, bs->pred);
        bs->mv = (vl264_mv){0, 0};
        bs->mb_type = MB_TYPE_P;

        // Quick zero-MV check: compute residual, DCT, quant
        int16_t zres[16];
        for (int i = 0; i < 16; i++) zres[i] = (int16_t)(orig[i] - bs->pred[i]);
        dct4x4_fwd(zres, zcoeff);
        quant4x4(zcoeff, qp, false);
        int32_t znz = 0;
        for (int i = 0; i < 16; i++) if (zcoeff[i] != 0) znz++;

        // Conditional ME: only search if zero-MV has many non-zero coefficients.
        // For 0-2 non-zero coefficients, the ME search rarely finds anything better.
        if (znz > 2) {
            int32_t best_sad = sad_4x4(orig, bs->pred);
            if (best_sad > 16) {
                int32_t me_sad;
                vl264_mv me_mv = me_search_int(orig, ref_slice, bx, by,
                                                (e->cfg.quality == VL264_FAST) ? 3 : 4, &me_sad);
                if (e->cfg.quality >= VL264_DEFAULT)
                    me_mv = me_refine_qpel(orig, ref_slice, bx, by, me_mv, &me_sad);
                if (me_sad < best_sad) {
                    bs->mv = me_mv;
                    mc_block(ref_slice, bx*4, by*4, bs->mv, bs->pred);
                }
            }
        }
    }

    // Residual, DCT, quant — skip if we already computed it for zero-MV
    int16_t residual[16];
    if (!is_iframe && ref_slice && bs->mv.x == 0 && bs->mv.y == 0) {
        memcpy(bs->coeff, zcoeff, sizeof(bs->coeff));
    } else {
        for (int i = 0; i < 16; i++) residual[i] = (int16_t)(orig[i] - bs->pred[i]);
        dct4x4_fwd(residual, bs->coeff);
        quant4x4(bs->coeff, qp, bs->mb_type == MB_TYPE_I4x4);
    }

    bs->total_coeff = 0;
    for (int i = 0; i < 16; i++) if (bs->coeff[i] != 0) bs->total_coeff++;

    // For skip to work, encoder and decoder must agree on prediction.
    // I-frame skip: decoder uses DC prediction → force DC here.
    // P-frame skip: decoder uses zero-motion ref → only skip if MV is zero.
    if (bs->total_coeff == 0) {
        if (bs->mb_type == MB_TYPE_I4x4) {
            int16_t top[4], left[4], tl;
            get_neighbors(recon_slice, bx, by, top, left, &tl);
            intra_pred_4x4(bs->pred, INTRA_DC, top, left, tl);
            bs->mode = INTRA_DC;
        } else if (bs->mb_type == MB_TYPE_P) {
            // P-frame skip: decoder uses zero-motion ref copy.
            // If MV is non-zero but coeffs are zero, fall back to zero-MV skip.
            if (bs->mv.x != 0 || bs->mv.y != 0) {
                // Revert to zero-MV prediction for skip
                get_block(ref_slice, bx*4, by*4, bs->pred);
                bs->mv = (vl264_mv){0, 0};
            }
        }
    }

    // Reconstruct
    int16_t dq[16], recon_res[16];
    memcpy(dq, bs->coeff, sizeof(dq));
    dequant4x4(dq, qp);
    dct4x4_inv(dq, recon_res);
    for (int dy = 0; dy < 4; dy++)
        for (int dx = 0; dx < 4; dx++) {
            int32_t v = bs->pred[dy*4+dx] + recon_res[dy*4+dx];
            recon_slice[(by*4+dy)*DIM + bx*4+dx] = (int16_t)VL264_CLAMP(v, 0, 255);
        }

    // Max error clamping: iteratively lower QP until error is within bounds
    if (e->cfg.max_error > 0 && qp > 1) {
        int32_t retry_qp = qp;
        for (int attempt = 0; attempt < 4 && retry_qp > 1; attempt++) {
            int32_t max_err = 0;
            for (int dy = 0; dy < 4; dy++)
                for (int dx = 0; dx < 4; dx++) {
                    int32_t err = abs(orig[dy*4+dx] - recon_slice[(by*4+dy)*DIM + bx*4+dx]);
                    if (err > max_err) max_err = err;
                }
            if (max_err <= e->cfg.max_error) break;
            retry_qp = VL264_MAX(retry_qp / 2, 1);
            for (int i = 0; i < 16; i++) residual[i] = (int16_t)(orig[i] - bs->pred[i]);
            dct4x4_fwd(residual, bs->coeff);
            quant4x4(bs->coeff, retry_qp, bs->mb_type == MB_TYPE_I4x4);
            bs->total_coeff = 0;
            for (int i = 0; i < 16; i++) if (bs->coeff[i] != 0) bs->total_coeff++;
            memcpy(dq, bs->coeff, sizeof(dq));
            dequant4x4(dq, retry_qp);
            dct4x4_inv(dq, recon_res);
            for (int dy = 0; dy < 4; dy++)
                for (int dx = 0; dx < 4; dx++) {
                    int32_t v2 = bs->pred[dy*4+dx] + recon_res[dy*4+dx];
                    recon_slice[(by*4+dy)*DIM + bx*4+dx] = (int16_t)VL264_CLAMP(v2, 0, 255);
                }
        }
    }

    return bs->total_coeff;
}

// MV prediction: median of left and above neighbors
VL264_INTERNAL vl264_mv predict_mv(const vl264_mv* mv_map, int32_t bx, int32_t by) {
    vl264_mv pred = {0, 0};
    if (bx > 0 && by > 0) {
        vl264_mv l = mv_map[by * BSTRIDE + bx - 1];
        vl264_mv t = mv_map[(by-1) * BSTRIDE + bx];
        pred.x = (int16_t)((l.x + t.x + 1) >> 1);
        pred.y = (int16_t)((l.y + t.y + 1) >> 1);
    } else if (bx > 0) {
        pred = mv_map[by * BSTRIDE + bx - 1];
    } else if (by > 0) {
        pred = mv_map[(by-1) * BSTRIDE + bx];
    }
    return pred;
}

// DC prediction: predict current block's DC from left/above reconstructed DC
VL264_INTERNAL int16_t predict_dc(const int16_t* dc_map, int32_t bx, int32_t by) {
    bool has_left = bx > 0;
    bool has_top = by > 0;
    if (has_left && has_top) {
        return (int16_t)((dc_map[by * BSTRIDE + bx - 1] + dc_map[(by-1) * BSTRIDE + bx] + 1) >> 1);
    } else if (has_left) {
        return dc_map[by * BSTRIDE + bx - 1];
    } else if (has_top) {
        return dc_map[(by-1) * BSTRIDE + bx];
    }
    return 0;
}

// Phase 2: Write a coded block to the bitstream.
VL264_INTERNAL void write_coded_block(vl264_enc* e, bs_writer* w,
                                       const block_state_t* bs,
                                       int32_t bx, int32_t by) {
    if (bs->mb_type == MB_TYPE_I4x4) {
        bs_w_bit1(w, 0); // intra
        if (bs->mode == INTRA_DC) {
            bs_w_bit1(w, 0);
        } else {
            bs_w_bit1(w, 1);
            int32_t idx = bs->mode < INTRA_DC ? bs->mode : bs->mode - 1;
            bs_w_bits(w, (uint32_t)idx, 3);
        }
    } else {
        bs_w_bit1(w, 1); // inter
        vl264_mv mvp = predict_mv(e->mv_map, bx, by);
        int16_t mvdx = (int16_t)(bs->mv.x - mvp.x);
        int16_t mvdy = (int16_t)(bs->mv.y - mvp.y);
        if (mvdx == 0 && mvdy == 0) {
            bs_w_bit1(w, 0); // predicted MV (1 bit)
        } else {
            bs_w_bit1(w, 1); // has MV delta
            bs_w_se(w, mvdx);
            bs_w_se(w, mvdy);
        }
    }

    // Apply DC prediction before writing coefficients
    int16_t coeff_out[16];
    memcpy(coeff_out, bs->coeff, sizeof(coeff_out));
    int16_t dc_pred = predict_dc(e->dc_map, bx, by);
    coeff_out[0] = (int16_t)(coeff_out[0] - dc_pred); // encode DC as delta

    int32_t nc = 0;
    if (bx > 0) nc += e->nc_map[by * BSTRIDE + bx - 1];
    if (by > 0) nc += e->nc_map[(by-1) * BSTRIDE + bx];
    if (bx > 0 && by > 0) nc = (nc + 1) / 2;
    cavlc_encode_block(w, coeff_out, nc);
}

// Early skip threshold: if SAD < threshold, skip without DCT/quant.
// Conservative at low QP, aggressive at high QP.
VL264_INTERNAL int32_t early_skip_threshold(int32_t qp) {
    if (qp <= 15) return 8;      // very conservative at low QP
    if (qp <= 25) return 16 + (qp - 15) * 3; // ramp up: 16-46
    return 46 + (qp - 25) * 8;   // aggressive at high QP: 46-254
}

// Encode a full 128x128 slice with skip-run encoding and DC prediction.
VL264_INTERNAL VL264_HOT void encode_slice(vl264_enc* e, bs_writer* w,
                                  const int16_t* cur, int16_t* recon,
                                  const int16_t* ref, bool is_iframe,
                                  int32_t qp_base) {
    memset(e->nc_map, 0, sizeof(e->nc_map));
    memset(e->dc_map, 0, sizeof(e->dc_map));
    memset(e->mv_map, 0, sizeof(e->mv_map));

    if (!is_iframe && ref) {
        memcpy(recon, ref, DIM * DIM * sizeof(int16_t));
    } else {
        memset(recon, 0, DIM * DIM * sizeof(int16_t));
    }

    int32_t skip_run = 0;
    int32_t early_thresh = early_skip_threshold(qp_base);

    for (int32_t by = 0; by < BSTRIDE; by++) {
        for (int32_t bx = 0; bx < BSTRIDE; bx++) {
            // Early skip for P-frames: if zero-MV SAD is very low, skip without DCT
            if (!is_iframe && ref) {
                int16_t orig[16], ref_blk[16];
                for (int dy = 0; dy < 4; dy++)
                    for (int dx = 0; dx < 4; dx++)
                        orig[dy*4+dx] = cur[(by*4+dy)*DIM + bx*4+dx];
                get_block(ref, bx*4, by*4, ref_blk);
                int32_t sad = sad_4x4(orig, ref_blk);
                // Check max error before allowing early skip
                bool allow_skip = (sad < early_thresh);
                if (allow_skip && e->cfg.max_error > 0) {
                    int32_t max_err = 0;
                    for (int i = 0; i < 16; i++) {
                        int32_t err = abs(orig[i] - ref_blk[i]);
                        if (err > max_err) max_err = err;
                    }
                    if (max_err > e->cfg.max_error) allow_skip = false;
                }
                if (allow_skip) {
                    skip_run++;
                    e->nc_map[by * BSTRIDE + bx] = 0;
                    e->dc_map[by * BSTRIDE + bx] = 0; // skip blocks have DC=0 in prediction chain
                    e->stats.skip_blocks++;
                    e->stats.total_blocks++;
                    e->stats.zero_coeff_blocks++;
                    continue;
                }
            }

            block_state_t bs;
            int32_t tc = try_encode_block(e, &bs, cur, recon, ref, bx, by, is_iframe, qp_base);

            if (tc == 0) {
                skip_run++;
                e->nc_map[by * BSTRIDE + bx] = 0;
                e->dc_map[by * BSTRIDE + bx] = 0; // skip: DC=0 in prediction chain
                e->stats.skip_blocks++;
                e->stats.total_blocks++;
                e->stats.zero_coeff_blocks++;
            } else {
                e->dc_map[by * BSTRIDE + bx] = bs.coeff[0]; // quantized DC for prediction

                bs_w_ue(w, (uint32_t)skip_run);
                skip_run = 0;
                write_coded_block(e, w, &bs, bx, by);
                e->mv_map[by * BSTRIDE + bx] = bs.mv;
                e->nc_map[by * BSTRIDE + bx] = (int16_t)(tc > 0 ? tc : 1);
                e->stats.total_blocks++;
                e->stats.avg_nonzero_coeffs += (float)VL264_MAX(tc, 0);
                if (bs.mb_type == MB_TYPE_I4x4) e->stats.intra_blocks++;
                else e->stats.inter_blocks++;
            }
        }
    }
    bs_w_ue(w, (uint32_t)skip_run);
}

// Helper: begin a new NAL unit. Writes 2-byte length placeholder + type byte.
VL264_INTERNAL size_t bs_w_nal_begin(bs_writer* w, uint8_t nal_type) {
    bs_w_flush(w);
    bs_w_align(w);
    bs_w_flush(w);
    size_t length_offset = w->byte_pos;
    if (w->byte_pos + NAL_LEN_BYTES + 1 <= w->capacity) {
        w->buf[w->byte_pos++] = 0; // length high byte placeholder
        w->buf[w->byte_pos++] = 0; // length low byte placeholder
        w->buf[w->byte_pos++] = nal_type & 0x1F;
    }
    w->cache = 0;
    w->bits_left = 32;
    return length_offset;
}

// Helper: finish a NAL unit, patching the length field.
VL264_INTERNAL void bs_w_nal_end(bs_writer* w, size_t length_offset) {
    bs_w_flush(w);
    bs_w_align(w);
    bs_w_flush(w);
    size_t nal_content_len = w->byte_pos - (length_offset + NAL_LEN_BYTES);
    if (nal_content_len > NAL_MAX_SIZE) nal_content_len = NAL_MAX_SIZE;
    w->buf[length_offset + 0] = (uint8_t)(nal_content_len >> 8);
    w->buf[length_offset + 1] = (uint8_t)(nal_content_len);
}

// Top-level encode
VL264_INTERNAL vl264_status encode_chunk_impl(vl264_enc* e,
                                               const uint8_t* input,
                                               const vl264_neighbors* neighbors,
                                               const vl264_lod_ref* lod,
                                               vl264_buf* out) {
    // 1. Select axis
    int32_t axis;
    if (e->cfg.axis == VL264_AXIS_AUTO)
        axis = select_best_axis(input);
    else
        axis = (int32_t)e->cfg.axis;

    // 2. Detect and apply bit-depth shift
    int32_t bit_shift = 0;
    if (e->cfg.bit_depth > 0 && e->cfg.bit_depth < 8) {
        bit_shift = 8 - e->cfg.bit_depth;
    } else if (e->cfg.bit_depth == 0) {
        bit_shift = detect_bit_shift(input);
    }
    // Apply right-shift to input (use a stack copy for small shifts, heap for safety)
    uint8_t* shifted_input = NULL;
    const uint8_t* enc_input = input;
    if (bit_shift > 0) {
        shifted_input = (uint8_t*)malloc(VL264_CHUNK_VOXELS);
        if (!shifted_input) return VL264_ERR_ALLOC;
        for (size_t i = 0; i < VL264_CHUNK_VOXELS; i++)
            shifted_input[i] = input[i] >> bit_shift;
        enc_input = shifted_input;
    }

    // 3. Set QP (user's QP respected directly)
    int32_t qp_base = e->cfg.qp;
    if (qp_base == 0) {
        switch (e->cfg.quality) {
        case VL264_FAST:    qp_base = 32; break;
        case VL264_DEFAULT: qp_base = 24; break;
        case VL264_MAX:     qp_base = 18; break;
        }
    }
    // chunk classification removed — user's QP is respected directly
    e->resolved_qp = qp_base;

    // 3. Generate slice ordering
    if (e->cfg.morton_order)
        morton_order_128(e->slice_order);
    else
        raster_order_128(e->slice_order);

    // 4. LOD delta: upsample coarse
    bool use_lod = e->cfg.lod_delta && lod && lod->data && lod->dim > 0 && lod->dim <= DIM;
    if (use_lod) {
        if (!e->lod_upsampled) {
            e->lod_upsampled = (int16_t*)aligned_alloc(VL264_ALIGN,
                (size_t)DIM * DIM * DIM * sizeof(int16_t));
            if (!e->lod_upsampled) return VL264_ERR_ALLOC;
        }
        upsample_2x(lod->data, lod->dim, e->lod_upsampled);
    }

    // 5. Allocate output if needed
    if (!out->data) {
        out->capacity = vl264_max_compressed_size();
        out->data = (uint8_t*)malloc(out->capacity);
        if (!out->data) return VL264_ERR_ALLOC;
    }

    // Single continuous writer for entire output
    bs_writer w;
    bs_w_init(&w, out->data, out->capacity);

    // 6. Write SPS NAL
    size_t sps_off = bs_w_nal_begin(&w, NAL_SPS);
    // Axis must be resolved by now (never AUTO)
    if (axis < 0 || axis > 2) return VL264_ERR_INVALID;
    vl264_sps_data sps = {
        .axis = (uint8_t)axis,
        .qp_base = (uint8_t)VL264_CLAMP(qp_base, 0, 51),
        .morton = e->cfg.morton_order ? 1 : 0,
        .has_boundary = (neighbors != NULL) && e->cfg.boundary_pred ? 1 : 0,
        .has_lod = use_lod ? 1 : 0,
        .bit_shift = (uint8_t)bit_shift,
    };
    write_sps(&w, &sps);
    bs_w_nal_end(&w, sps_off);

    // 7. Write PPS NAL
    size_t pps_off = bs_w_nal_begin(&w, NAL_PPS);
    write_pps(&w);
    bs_w_nal_end(&w, pps_off);

    // 8. Encode slices
    double t_start = vl264_clock_sec();
    memset(&e->stats, 0, sizeof(e->stats));
    e->stats.axis = (vl264_axis)axis;
    memset(e->ref_slice, 0, sizeof(e->ref_slice));

    VL264_ALIGNED(64) int16_t lod_pred_slice[DIM * DIM];

    for (uint32_t ci = 0; ci < DIM; ci++) {
        uint32_t spatial_idx = e->slice_order[ci];

        // Extract slice
        VL264_ALIGNED(64) int16_t cur[DIM * DIM];
        extract_slice(enc_input, axis, (int32_t)spatial_idx, cur);

        // LOD delta: subtract upsampled prediction
        if (use_lod) {
            lod_extract_slice(e->lod_upsampled, axis, (int32_t)spatial_idx, lod_pred_slice);
            for (int32_t i = 0; i < DIM * DIM; i++) {
                cur[i] = (int16_t)(cur[i] - lod_pred_slice[i] + 128);
            }
        }

        // Determine I or P frame
        int32_t iframe_interval = e->cfg.iframe_interval;
        if (iframe_interval <= 0) {
            switch (e->cfg.quality) {
            case VL264_MAX:     iframe_interval = 16; break;
            case VL264_DEFAULT: iframe_interval = 32; break;
            default:            iframe_interval = 64; break;
            }
        }
        bool is_iframe = should_force_iframe(e->slice_order, ci, iframe_interval);

        // Select reference
        const int16_t* ref = is_iframe ? NULL : e->ref_slice;
        if (!is_iframe && e->cfg.boundary_pred && neighbors) {
            const uint8_t* face = get_boundary_face(neighbors, axis, (int32_t)spatial_idx);
            if (face && (spatial_idx == 0 || spatial_idx == DIM - 1)) {
                widen_face(face, e->boundary_buf);
                ref = e->boundary_buf;
            }
        }

        // Write slice NAL
        size_t slice_off = bs_w_nal_begin(&w, is_iframe ? NAL_SLICE_IDR : NAL_SLICE_P);

        // Write slice header
        vl264_slice_hdr hdr = {
            .slice_idx = spatial_idx,
            .is_idr = is_iframe ? 1 : 0,
            .qp_delta = 0,
            .ref_idx = ci > 0 ? e->slice_order[ci - 1] : 0,
        };
        write_slice_hdr(&w, &hdr);

        // Encode all blocks
        memset(e->recon_slice, 0, sizeof(e->recon_slice));
        if (!is_iframe && ref) {
            memcpy(e->recon_slice, ref, sizeof(e->recon_slice));
        }

        encode_slice(e, &w, cur, e->recon_slice, ref, is_iframe, qp_base);
        bs_w_nal_end(&w, slice_off);

        // Store reconstructed as reference
        memcpy(e->ref_slice, e->recon_slice, sizeof(e->ref_slice));

        if (is_iframe) e->stats.i_slices++;
        else e->stats.p_slices++;
    }

    bs_w_flush(&w);
    out->size = w.byte_pos;

    // Stats
    double t_end = vl264_clock_sec();
    e->stats.encode_sec = t_end - t_start;
    e->stats.input_bytes = VL264_CHUNK_VOXELS;
    e->stats.output_bytes = out->size;
    e->stats.ratio = (out->size > 0) ? (float)VL264_CHUNK_VOXELS / (float)out->size : 0.0f;
    e->stats.bits_per_voxel = (out->size > 0) ? (float)out->size * 8.0f / (float)VL264_CHUNK_VOXELS : 0.0f;
    e->stats.avg_qp = (float)qp_base;
    if (e->stats.encode_sec > 0.0)
        e->stats.encode_mbs = (double)VL264_CHUNK_VOXELS / (e->stats.encode_sec * 1e6);
    if (e->stats.total_blocks > 0)
        e->stats.avg_nonzero_coeffs /= (float)e->stats.total_blocks;

    free(shifted_input);
    return VL264_OK;
}

// ═════════════════════════════════════════════════════════════════════════════
// Section 19: Decoder state + core
// ═════════════════════════════════════════════════════════════════════════════

struct vl264_dec {
    int32_t axis;
    int32_t qp_base;
    bool    has_boundary;
    bool    has_lod;
    bool    morton;

    VL264_ALIGNED(64) int16_t ref_slice[DIM * DIM];
    VL264_ALIGNED(64) int16_t recon_slice[DIM * DIM];
    VL264_ALIGNED(64) int16_t boundary_buf[DIM * DIM];

    int16_t* lod_upsampled;

    int16_t nc_map[BSTRIDE * BSTRIDE];
    int16_t dc_map[BSTRIDE * BSTRIDE];
    vl264_mv mv_map[BSTRIDE * BSTRIDE];
    uint32_t slice_order[DIM];

    // Streaming state
    const uint8_t* bs_data;
    size_t         bs_size;
    size_t         bs_offset;
    int32_t        next_slice; // -1 = not started
    vl264_sps_data sps;

    // Cached context
    const vl264_neighbors* neighbors;
    const vl264_lod_ref*   lod;
};

// Decode a single coded 4x4 block (skip handling is in decode_slice_blocks)
// Format: bit1(type) + [4-bit mode | se(mvx)+se(mvy)] + coefficients
VL264_INTERNAL VL264_HOT void decode_block(vl264_dec* d, bs_reader* r,
                                  int16_t* recon_slice, const int16_t* ref_slice,
                                  int32_t bx, int32_t by, int32_t qp_base) {
    int32_t qp = qp_base;
    uint32_t is_inter = bs_r_bit1(r);

    int32_t mode = 0;
    vl264_mv mv = {0, 0};

    if (!is_inter) {
        if (bs_r_bit1(r) == 0) {
            mode = INTRA_DC;
        } else {
            int32_t idx = (int32_t)bs_r_bits(r, 3);
            mode = idx < INTRA_DC ? idx : idx + 1;
            if (mode >= INTRA_NUM_MODES) mode = INTRA_DC;
        }
    } else {
        vl264_mv mvp = predict_mv(d->mv_map, bx, by);
        if (bs_r_bit1(r)) { // has MV delta
            mv.x = (int16_t)(mvp.x + bs_r_se(r));
            mv.y = (int16_t)(mvp.y + bs_r_se(r));
        } else {
            mv = mvp; // predicted MV
        }
        d->mv_map[by * BSTRIDE + bx] = mv;
    }

    // Decode coefficients
    int32_t nc = 0;
    if (bx > 0) nc += d->nc_map[by * BSTRIDE + bx - 1];
    if (by > 0) nc += d->nc_map[(by-1) * BSTRIDE + bx];
    if (bx > 0 && by > 0) nc = (nc + 1) / 2;

    int16_t coeff[16];
    cavlc_decode_block(r, coeff, nc);

    // Undo DC prediction: add back predicted DC
    int16_t dc_pred = predict_dc(d->dc_map, bx, by);
    coeff[0] = (int16_t)(coeff[0] + dc_pred);

    int32_t total_coeff = 0;
    for (int i = 0; i < 16; i++) if (coeff[i] != 0) total_coeff++;
    d->nc_map[by * BSTRIDE + bx] = (int16_t)total_coeff;
    d->dc_map[by * BSTRIDE + bx] = coeff[0]; // store for next block's prediction

    // Dequant + inverse DCT
    dequant4x4(coeff, qp);
    int16_t residual[16];
    dct4x4_inv(coeff, residual);

    // Prediction
    int16_t pred[16];
    if (!is_inter) {
        int16_t top[4], left[4], tl;
        get_neighbors(recon_slice, bx, by, top, left, &tl);
        intra_pred_4x4(pred, mode, top, left, tl);
    } else {
        const int16_t* mc_ref = ref_slice ? ref_slice : recon_slice;
        mc_block(mc_ref, bx * 4, by * 4, mv, pred);
    }

    // Reconstruct
    for (int dy = 0; dy < 4; dy++)
        for (int dx = 0; dx < 4; dx++) {
            int32_t v = pred[dy*4+dx] + residual[dy*4+dx];
            recon_slice[(by*4+dy) * DIM + bx*4+dx] = (int16_t)VL264_CLAMP(v, 0, 255);
        }
}

// Decode all blocks in a slice.
// Format: skip_run encoding. Read ue(skip_run), skip that many blocks,
// then decode one coded block. Repeat until all 1024 blocks processed.
VL264_INTERNAL VL264_HOT void decode_slice_blocks(vl264_dec* d, bs_reader* r,
                                         int16_t* recon, const int16_t* ref,
                                         int32_t qp, bool is_iframe) {
    memset(d->nc_map, 0, sizeof(d->nc_map));
    memset(d->dc_map, 0, sizeof(d->dc_map));
    memset(d->mv_map, 0, sizeof(d->mv_map));

    if (!is_iframe && ref) {
        memcpy(recon, ref, DIM * DIM * sizeof(int16_t));
    } else {
        memset(recon, 0, DIM * DIM * sizeof(int16_t));
    }

    int32_t total_blocks = BSTRIDE * BSTRIDE;
    int32_t block_idx = 0;

    while (block_idx < total_blocks) {
        uint32_t skip_run = bs_r_ue(r);
        if (skip_run > (uint32_t)(total_blocks - block_idx))
            skip_run = (uint32_t)(total_blocks - block_idx);
        // Skip `skip_run` blocks (prediction already in recon)
        for (uint32_t s = 0; s < skip_run && block_idx < total_blocks; s++, block_idx++) {
            int32_t bx = block_idx % BSTRIDE;
            int32_t by = block_idx / BSTRIDE;
            // For I-frame: need to compute prediction for skipped blocks
            if (is_iframe || !ref) {
                int16_t top[4], left[4], tl, pred[16];
                get_neighbors(recon, bx, by, top, left, &tl);
                intra_pred_4x4(pred, INTRA_DC, top, left, tl);
                for (int dy = 0; dy < 4; dy++)
                    for (int dx = 0; dx < 4; dx++)
                        recon[(by*4+dy)*DIM + bx*4+dx] = (int16_t)VL264_CLAMP(pred[dy*4+dx], 0, 255);
            }
            // P-frame skip: recon already has reference data
            d->nc_map[by * BSTRIDE + bx] = 0;
        }

        if (block_idx >= total_blocks) break;

        // Decode one coded block
        // The coded data was written as whole bytes from a temp buffer,
        // so we read it byte-aligned from the stream.
        int32_t bx = block_idx % BSTRIDE;
        int32_t by = block_idx / BSTRIDE;

        // Read coded block data (written as bytes, so align first)
        // Actually no — the bytes were written via bs_w_bits(w, byte, 8) which
        // doesn't necessarily byte-align. But since each block's temp buffer
        // was flushed, the bytes are complete. We need to read them the same way.
        // The simplest: read a byte, feed to a temp reader, decode block from that.
        // But that adds complexity. Instead: just call decode_block which reads
        // bit by bit — same format.
        decode_block(d, r, recon, ref, bx, by, qp);
        block_idx++;
    }
}

// Full chunk decode
VL264_INTERNAL vl264_status decode_chunk_impl(vl264_dec* d,
                                               const uint8_t* bitstream,
                                               size_t bitstream_size,
                                               const vl264_neighbors* neighbors,
                                               const vl264_lod_ref* lod,
                                               uint8_t* chunk_out) {
    // Find NAL units
    size_t offsets[256], sizes[256];
    size_t nal_count = find_nal_units(bitstream, bitstream_size, offsets, sizes, 256);
    if (nal_count < 2) return VL264_ERR_CORRUPT;

    // Parse SPS (first NAL after start code)
    bs_reader r;
    bs_r_init(&r, bitstream + offsets[0] + 1, sizes[0] - 1); // skip NAL type byte
    read_sps(&r, &d->sps);
    d->axis = d->sps.axis;
    d->qp_base = d->sps.qp_base;
    d->morton = d->sps.morton;
    d->has_boundary = d->sps.has_boundary;
    d->has_lod = d->sps.has_lod;

    // Parse PPS (second NAL)
    bs_r_init(&r, bitstream + offsets[1] + 1, sizes[1] - 1);
    read_pps(&r);

    // Generate slice ordering
    if (d->morton)
        morton_order_128(d->slice_order);
    else
        raster_order_128(d->slice_order);

    // LOD upsampling
    bool use_lod = d->has_lod && lod && lod->data && lod->dim > 0;
    if (use_lod) {
        if (!d->lod_upsampled) {
            d->lod_upsampled = (int16_t*)aligned_alloc(VL264_ALIGN,
                (size_t)DIM * DIM * DIM * sizeof(int16_t));
            if (!d->lod_upsampled) return VL264_ERR_ALLOC;
        }
        upsample_2x(lod->data, lod->dim, d->lod_upsampled);
    }

    // Initialize reference
    memset(d->ref_slice, 0, sizeof(d->ref_slice));

    // Initialize output
    memset(chunk_out, 0, VL264_CHUNK_VOXELS);

    VL264_ALIGNED(64) int16_t lod_pred_slice[DIM * DIM];

    // Decode slices (NAL units 2..nal_count-1)
    uint32_t slice_ci = 0;
    for (size_t ni = 2; ni < nal_count && slice_ci < DIM; ni++) {
        const uint8_t* nal_data = bitstream + offsets[ni];
        size_t nal_size = sizes[ni];
        if (nal_size < 2) continue;

        uint8_t nal_type = nal_data[0] & 0x1F;
        if (nal_type != NAL_SLICE_IDR && nal_type != NAL_SLICE_P) continue;

        bs_r_init(&r, nal_data + 1, nal_size - 1);

        // Read slice header
        vl264_slice_hdr hdr;
        read_slice_hdr(&r, &hdr);

        bool is_iframe = hdr.is_idr;
        uint32_t spatial_idx = hdr.slice_idx;

        // Select reference
        const int16_t* ref = is_iframe ? NULL : d->ref_slice;
        if (!is_iframe && d->has_boundary && neighbors) {
            const uint8_t* face = get_boundary_face(neighbors, d->axis, (int32_t)spatial_idx);
            if (face && (spatial_idx == 0 || spatial_idx == DIM - 1)) {
                widen_face(face, d->boundary_buf);
                ref = d->boundary_buf;
            }
        }

        // Decode slice
        decode_slice_blocks(d, &r, d->recon_slice, ref, d->qp_base, is_iframe);

        // LOD delta: add back prediction
        if (use_lod) {
            lod_extract_slice(d->lod_upsampled, d->axis, (int32_t)spatial_idx, lod_pred_slice);
            for (int32_t i = 0; i < DIM * DIM; i++) {
                d->recon_slice[i] = (int16_t)(d->recon_slice[i] - 128 + lod_pred_slice[i]);
            }
        }

        // Insert into output volume
        insert_slice(chunk_out, d->axis, (int32_t)spatial_idx, d->recon_slice);

        // Store as reference
        memcpy(d->ref_slice, d->recon_slice, sizeof(d->ref_slice));
        slice_ci++;
    }

    // Apply left-shift to restore original bit depth
    if (d->sps.bit_shift > 0) {
        for (size_t i = 0; i < VL264_CHUNK_VOXELS; i++)
            chunk_out[i] = (uint8_t)(chunk_out[i] << d->sps.bit_shift);
    }

    return VL264_OK;
}

// ═════════════════════════════════════════════════════════════════════════════
// Section 20: Public API
// ═════════════════════════════════════════════════════════════════════════════

VL264_COLD const char* vl264_status_str(vl264_status s) {
    switch (s) {
    case VL264_OK:            return "ok";
    case VL264_ERR_NULL_ARG:  return "null argument";
    case VL264_ERR_ALLOC:     return "allocation failed";
    case VL264_ERR_BITSTREAM: return "bitstream error";
    case VL264_ERR_CORRUPT:   return "corrupt data";
    case VL264_ERR_OVERFLOW:  return "overflow";
    case VL264_ERR_INVALID:   return "invalid parameter";
    }
    return "unknown";
}

const char* vl264_version_str(void) {
    return "vl264 0.1.0";
}

vl264_cfg vl264_default_cfg(void) {
    return (vl264_cfg){
        .quality         = VL264_DEFAULT,
        .qp              = 0, // auto
        .qp_sensitivity  = 0.6f,
        .axis            = VL264_AXIS_AUTO,
        .boundary_pred   = false,
        .lod_delta       = false,
        .morton_order    = false,
        .iframe_interval = 0, // auto from quality
        .max_error       = 0, // unlimited
        .bit_depth       = 0, // auto-detect
    };
}

size_t vl264_max_compressed_size(void) {
    // Worst case: original size + overhead for headers/NAL framing
    return VL264_CHUNK_VOXELS * 2 + 4096;
}

void vl264_free(void* ptr) {
    free(ptr);
}

// Encoder

vl264_enc* vl264_enc_create(const vl264_cfg* cfg) {
    vl264_enc* e = (vl264_enc*)aligned_alloc(VL264_ALIGN, sizeof(vl264_enc));
    if (!e) return NULL;
    memset(e, 0, sizeof(*e));
    e->cfg = cfg ? *cfg : vl264_default_cfg();
    e->lod_upsampled = NULL;
    return e;
}

void vl264_enc_destroy(vl264_enc* e) {
    if (!e) return;
    free(e->lod_upsampled);
    free(e);
}

vl264_status vl264_encode(vl264_enc* e, const uint8_t* chunk,
                           const vl264_neighbors* neighbors,
                           const vl264_lod_ref* lod, vl264_buf* out) {
    if (!e || !chunk || !out) return VL264_ERR_NULL_ARG;
    // Validate output buffer: if pre-allocated, must be large enough
    if (out->data && out->capacity < vl264_max_compressed_size())
        return VL264_ERR_OVERFLOW;
    // Validate QP
    if (e->cfg.qp < 0 || e->cfg.qp > 51) return VL264_ERR_INVALID;
    return encode_chunk_impl(e, chunk, neighbors, lod, out);
}

vl264_status vl264_enc_stats_get(const vl264_enc* e, vl264_enc_stats* s) {
    if (!e || !s) return VL264_ERR_NULL_ARG;
    *s = e->stats;
    return VL264_OK;
}

// Decoder

vl264_dec* vl264_dec_create(void) {
    vl264_dec* d = (vl264_dec*)aligned_alloc(VL264_ALIGN, sizeof(vl264_dec));
    if (!d) return NULL;
    memset(d, 0, sizeof(*d));
    d->next_slice = -1;
    return d;
}

void vl264_dec_destroy(vl264_dec* d) {
    if (!d) return;
    free(d->lod_upsampled);
    free(d);
}

vl264_status vl264_decode(vl264_dec* d, const uint8_t* bitstream, size_t bitstream_size,
                           const vl264_neighbors* neighbors, const vl264_lod_ref* lod,
                           uint8_t* chunk_out) {
    if (!d || !bitstream || !chunk_out || bitstream_size == 0) return VL264_ERR_NULL_ARG;
    return decode_chunk_impl(d, bitstream, bitstream_size, neighbors, lod, chunk_out);
}

vl264_status vl264_decode_begin(vl264_dec* d, const uint8_t* bitstream, size_t bitstream_size,
                                 const vl264_neighbors* neighbors, const vl264_lod_ref* lod) {
    if (!d || !bitstream) return VL264_ERR_NULL_ARG;
    d->bs_data = bitstream;
    d->bs_size = bitstream_size;
    d->neighbors = neighbors;
    d->lod = lod;
    d->next_slice = 0;

    // Parse headers
    size_t offsets[256], sizes_arr[256];
    size_t nal_count = find_nal_units(bitstream, bitstream_size, offsets, sizes_arr, 256);
    if (nal_count < 2) return VL264_ERR_CORRUPT;

    bs_reader r;
    bs_r_init(&r, bitstream + offsets[0] + 1, sizes_arr[0] - 1);
    read_sps(&r, &d->sps);
    d->axis = d->sps.axis;
    d->qp_base = d->sps.qp_base;
    d->morton = d->sps.morton;
    d->has_boundary = d->sps.has_boundary;
    d->has_lod = d->sps.has_lod;

    if (d->morton)
        morton_order_128(d->slice_order);
    else
        raster_order_128(d->slice_order);

    bool use_lod = d->has_lod && lod && lod->data && lod->dim > 0;
    if (use_lod) {
        if (!d->lod_upsampled) {
            d->lod_upsampled = (int16_t*)aligned_alloc(VL264_ALIGN,
                (size_t)DIM * DIM * DIM * sizeof(int16_t));
            if (!d->lod_upsampled) return VL264_ERR_ALLOC;
        }
        upsample_2x(lod->data, lod->dim, d->lod_upsampled);
    }

    d->bs_offset = offsets[1] + sizes_arr[1]; // skip SPS + PPS
    memset(d->ref_slice, 0, sizeof(d->ref_slice));

    return VL264_OK;
}

vl264_status vl264_decode_next_slice(vl264_dec* d, uint8_t* slice_out, uint32_t* slice_index) {
    if (!d || !slice_out || !slice_index) return VL264_ERR_NULL_ARG;
    if (d->next_slice < 0) return VL264_ERR_INVALID;
    if (d->next_slice >= DIM) return VL264_ERR_OVERFLOW;

    // Find next slice NAL from current offset
    size_t offsets[4], sizes_arr[4];
    size_t remaining = d->bs_size - d->bs_offset;
    if (remaining < 5) return VL264_ERR_OVERFLOW;

    size_t count = find_nal_units(d->bs_data + d->bs_offset, remaining, offsets, sizes_arr, 1);
    if (count == 0) return VL264_ERR_OVERFLOW;

    const uint8_t* nal_data = d->bs_data + d->bs_offset + offsets[0];
    size_t nal_size = sizes_arr[0];

    bs_reader r;
    bs_r_init(&r, nal_data + 1, nal_size - 1);

    vl264_slice_hdr hdr;
    read_slice_hdr(&r, &hdr);

    bool is_iframe = hdr.is_idr;
    uint32_t spatial_idx = hdr.slice_idx;

    const int16_t* ref = is_iframe ? NULL : d->ref_slice;
    if (!is_iframe && d->has_boundary && d->neighbors) {
        const uint8_t* face = get_boundary_face(d->neighbors, d->axis, (int32_t)spatial_idx);
        if (face && (spatial_idx == 0 || spatial_idx == DIM - 1)) {
            widen_face(face, d->boundary_buf);
            ref = d->boundary_buf;
        }
    }

    decode_slice_blocks(d, &r, d->recon_slice, ref, d->qp_base, is_iframe);

    // LOD delta
    if (d->has_lod && d->lod_upsampled) {
        VL264_ALIGNED(64) int16_t lod_pred[DIM * DIM];
        lod_extract_slice(d->lod_upsampled, d->axis, (int32_t)spatial_idx, lod_pred);
        for (int32_t i = 0; i < DIM * DIM; i++) {
            d->recon_slice[i] = (int16_t)(d->recon_slice[i] - 128 + lod_pred[i]);
        }
    }

    // Clamp and output
    clamp_i16_u8(slice_out, d->recon_slice, DIM * DIM);
    *slice_index = spatial_idx;

    memcpy(d->ref_slice, d->recon_slice, sizeof(d->ref_slice));
    d->bs_offset += offsets[0] + sizes_arr[0];
    d->next_slice++;

    return VL264_OK;
}

vl264_axis vl264_decode_axis(const vl264_dec* d) {
    return d ? (vl264_axis)d->axis : VL264_AXIS_Z;
}

// Utilities

vl264_axis vl264_analyze_axis(const uint8_t* chunk) {
    if (!chunk) return VL264_AXIS_Z;
    return (vl264_axis)select_best_axis(chunk);
}

float vl264_mse(const uint8_t* a, const uint8_t* b, size_t n) {
    if (!a || !b || n == 0) return 0.0f;
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        sum += d * d;
    }
    return (float)(sum / (double)n);
}

float vl264_psnr(const uint8_t* a, const uint8_t* b, size_t n) {
    float mse = vl264_mse(a, b, n);
    if (mse < 0.0001f) return INFINITY;
    return 10.0f * log10f(255.0f * 255.0f / mse);
}

// ── Comparison function for qsort ───────────────────────────────────────────

static int cmp_u32(const void* a, const void* b) {
    uint32_t va = *(const uint32_t*)a;
    uint32_t vb = *(const uint32_t*)b;
    return (va > vb) - (va < vb);
}

// ── Comprehensive stats computation ─────────────────────────────────────────

void vl264_stats_compute(vl264_stats* s, const uint8_t* original,
                          const uint8_t* decoded, size_t n) {
    if (!s || !original || !decoded || n == 0) return;

    // ── Error distribution ──────────────────────────────────────────────
    double sum_se = 0.0, sum_ae = 0.0;
    uint32_t max_ae = 0;

    // Histogram of absolute errors (0..255)
    uint32_t err_hist[256] = {0};

    for (size_t i = 0; i < n; i++) {
        int32_t d = (int32_t)original[i] - (int32_t)decoded[i];
        uint32_t ae = (uint32_t)abs(d);
        sum_se += (double)d * (double)d;
        sum_ae += (double)ae;
        if (ae > max_ae) max_ae = ae;
        if (ae < 256) err_hist[ae]++;
    }

    s->mse = (float)(sum_se / (double)n);
    s->mae = (float)(sum_ae / (double)n);
    s->max_err = (float)max_ae;
    s->psnr = (s->mse < 0.0001f) ? INFINITY : 10.0f * log10f(255.0f * 255.0f / s->mse);

    // Percentiles from histogram (O(256) instead of sorting O(n log n))
    size_t p50_idx = n / 2;
    size_t p90_idx = (size_t)((double)n * 0.90);
    size_t p95_idx = (size_t)((double)n * 0.95);
    size_t p99_idx = (size_t)((double)n * 0.99);
    size_t cumulative = 0;
    s->p50_err = s->p90_err = s->p95_err = s->p99_err = (float)max_ae;

    for (uint32_t e = 0; e < 256; e++) {
        cumulative += err_hist[e];
        if (cumulative >= p50_idx && s->p50_err == (float)max_ae) s->p50_err = (float)e;
        if (cumulative >= p90_idx && s->p90_err == (float)max_ae) s->p90_err = (float)e;
        if (cumulative >= p95_idx && s->p95_err == (float)max_ae) s->p95_err = (float)e;
        if (cumulative >= p99_idx && s->p99_err == (float)max_ae) s->p99_err = (float)e;
    }

    // ── Input analysis ──────────────────────────────────────────────────
    double in_sum = 0.0, in_sum2 = 0.0;
    uint8_t in_min = 255, in_max = 0;
    uint32_t in_hist[256] = {0};

    for (size_t i = 0; i < n; i++) {
        uint8_t v = original[i];
        in_sum += v;
        in_sum2 += (double)v * v;
        if (v < in_min) in_min = v;
        if (v > in_max) in_max = v;
        in_hist[v]++;
    }

    s->input_mean = (float)(in_sum / (double)n);
    double var = in_sum2 / (double)n - (double)s->input_mean * (double)s->input_mean;
    s->input_stddev = (float)sqrt(var > 0.0 ? var : 0.0);
    s->input_min = in_min;
    s->input_max = in_max;

    // Shannon entropy
    double entropy = 0.0;
    for (int v = 0; v < 256; v++) {
        if (in_hist[v] > 0) {
            double p = (double)in_hist[v] / (double)n;
            entropy -= p * log2(p);
        }
    }
    s->input_entropy = (float)entropy;

    // ── Compression fields ──────────────────────────────────────────────
    s->input_bytes = n;
    if (s->output_bytes > 0) {
        s->ratio = (float)n / (float)s->output_bytes;
        s->bits_per_voxel = (float)s->output_bytes * 8.0f / (float)n;
    }
}

VL264_COLD void vl264_stats_print(const vl264_stats* s, FILE* out) {
    if (!s || !out) return;

    fprintf(out, "┌─────────────────────────────────────────────────┐\n");
    fprintf(out, "│  VL264 Statistics Report                        │\n");
    fprintf(out, "├─────────────────────────────────────────────────┤\n");

    fprintf(out, "│ Compression                                     │\n");
    fprintf(out, "│   Input:         %10zu bytes                │\n", s->input_bytes);
    fprintf(out, "│   Output:        %10zu bytes                │\n", s->output_bytes);
    fprintf(out, "│   Ratio:         %10.2f:1                   │\n", s->ratio);
    fprintf(out, "│   Bits/voxel:    %10.3f                      │\n", s->bits_per_voxel);
    fprintf(out, "│   Axis:          %10d                      │\n", s->axis);

    fprintf(out, "├─────────────────────────────────────────────────┤\n");
    fprintf(out, "│ Timing                                          │\n");
    fprintf(out, "│   Encode:        %10.3f ms                  │\n", s->encode_sec * 1000.0);
    fprintf(out, "│   Decode:        %10.3f ms                  │\n", s->decode_sec * 1000.0);
    fprintf(out, "│   Enc throughput: %9.1f MB/s                │\n", s->encode_mbs);
    fprintf(out, "│   Dec throughput: %9.1f MB/s                │\n", s->decode_mbs);

    fprintf(out, "├─────────────────────────────────────────────────┤\n");
    fprintf(out, "│ Quality                                         │\n");
    fprintf(out, "│   PSNR:          %10.2f dB                  │\n", s->psnr);
    fprintf(out, "│   MSE:           %10.4f                      │\n", s->mse);
    fprintf(out, "│   MAE:           %10.4f                      │\n", s->mae);
    fprintf(out, "│   Max error:     %10.1f                      │\n", s->max_err);
    fprintf(out, "│   P50 error:     %10.1f                      │\n", s->p50_err);
    fprintf(out, "│   P90 error:     %10.1f                      │\n", s->p90_err);
    fprintf(out, "│   P95 error:     %10.1f                      │\n", s->p95_err);
    fprintf(out, "│   P99 error:     %10.1f                      │\n", s->p99_err);

    fprintf(out, "├─────────────────────────────────────────────────┤\n");
    fprintf(out, "│ Coding                                          │\n");
    fprintf(out, "│   Avg QP:        %10.1f                      │\n", s->avg_qp);
    fprintf(out, "│   I-slices:      %10u                      │\n", s->i_slices);
    fprintf(out, "│   P-slices:      %10u                      │\n", s->p_slices);
    fprintf(out, "│   Total blocks:  %10u                      │\n", s->total_blocks);
    fprintf(out, "│   Skip blocks:   %10u (%5.1f%%)             │\n",
            s->skip_blocks,
            s->total_blocks > 0 ? 100.0f * (float)s->skip_blocks / (float)s->total_blocks : 0.0f);
    fprintf(out, "│   Intra blocks:  %10u (%5.1f%%)             │\n",
            s->intra_blocks,
            s->total_blocks > 0 ? 100.0f * (float)s->intra_blocks / (float)s->total_blocks : 0.0f);
    fprintf(out, "│   Inter blocks:  %10u (%5.1f%%)             │\n",
            s->inter_blocks,
            s->total_blocks > 0 ? 100.0f * (float)s->inter_blocks / (float)s->total_blocks : 0.0f);
    fprintf(out, "│   Zero-coeff:    %10u (%5.1f%%)             │\n",
            s->zero_coeff_blocks,
            s->total_blocks > 0 ? 100.0f * (float)s->zero_coeff_blocks / (float)s->total_blocks : 0.0f);
    fprintf(out, "│   Avg NZ coeffs: %10.2f                      │\n", s->avg_nonzero_coeffs);

    fprintf(out, "├─────────────────────────────────────────────────┤\n");
    fprintf(out, "│ Input Analysis                                  │\n");
    fprintf(out, "│   Mean:          %10.2f                      │\n", s->input_mean);
    fprintf(out, "│   Stddev:        %10.2f                      │\n", s->input_stddev);
    fprintf(out, "│   Range:         [%3u, %3u]                     │\n", s->input_min, s->input_max);
    fprintf(out, "│   Entropy:       %10.3f bits/voxel           │\n", s->input_entropy);
    fprintf(out, "└─────────────────────────────────────────────────┘\n");
}
