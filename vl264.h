// vl264.h — Volume Librarian 264
// Pure C23 codec for 128x128x128 u8 volumetric CT data chunks.
// H.264 Baseline-derived with volume-specific optimizations.
// Zero dependencies beyond libc.
#ifndef VL264_H
#define VL264_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

// ── Version ─────────────────────────────────────────────────────────────────

#define VL264_VERSION_MAJOR 0
#define VL264_VERSION_MINOR 1
#define VL264_VERSION_PATCH 0

// ── Constants ───────────────────────────────────────────────────────────────

#define VL264_CHUNK_DIM    128
#define VL264_CHUNK_VOXELS (128 * 128 * 128) // 2,097,152
#define VL264_SLICE_DIM    128
#define VL264_SLICE_PIXELS (128 * 128) // 16,384
#define VL264_ALIGN        64          // SIMD / cache-line alignment

// ── Error codes ─────────────────────────────────────────────────────────────

typedef enum vl264_status {
    VL264_OK = 0,
    VL264_ERR_NULL_ARG,
    VL264_ERR_ALLOC,
    VL264_ERR_BITSTREAM,
    VL264_ERR_CORRUPT,
    VL264_ERR_OVERFLOW,
    VL264_ERR_INVALID,
} vl264_status;

const char* vl264_status_str(vl264_status s);

// ── Axis ────────────────────────────────────────────────────────────────────

typedef enum vl264_axis {
    VL264_AXIS_AUTO = -1, // pick axis with highest inter-slice correlation
    VL264_AXIS_X   = 0,
    VL264_AXIS_Y   = 1,
    VL264_AXIS_Z   = 2,
} vl264_axis;

// ── Quality presets ─────────────────────────────────────────────────────────

typedef enum vl264_quality {
    VL264_FAST,    // DC+planar intra only, high QP, integer-pel ME
    VL264_DEFAULT, // restricted intra + fallback, adaptive QP, half-pel
    VL264_MAX,     // all intra modes, fine adaptive QP, quarter-pel
} vl264_quality;

// ── Configuration ───────────────────────────────────────────────────────────

typedef struct vl264_cfg {
    vl264_quality quality;        // encoding quality preset
    int32_t       qp;            // base QP (0 = auto from quality, else 1-51)
    float         qp_sensitivity; // adaptive QP strength [0..1], default 0.6
    vl264_axis    axis;          // encode axis (VL264_AXIS_AUTO recommended)
    bool          boundary_pred; // inter-chunk boundary prediction
    bool          lod_delta;     // encode as residual vs upsampled coarser LOD
    bool          morton_order;  // space-filling curve slice ordering
} vl264_cfg;

// Returns a default configuration (VL264_DEFAULT quality).
vl264_cfg vl264_default_cfg(void);

// ── Neighbor context (inter-chunk boundary prediction) ──────────────────────

// Up to 6 boundary faces from already-decoded neighbor chunks.
// Each face is 128x128 u8. NULL entries = neighbor not available.
typedef struct vl264_neighbors {
    const uint8_t* neg_x; // x=0   face (neighbor chunk's x=127 slice)
    const uint8_t* pos_x; // x=127 face (neighbor chunk's x=0   slice)
    const uint8_t* neg_y; // y=0   face
    const uint8_t* pos_y; // y=127 face
    const uint8_t* neg_z; // z=0   face
    const uint8_t* pos_z; // z=127 face
} vl264_neighbors;

// ── LOD context (delta coding) ──────────────────────────────────────────────

// Coarser LOD chunk for hierarchical delta coding.
typedef struct vl264_lod_ref {
    const uint8_t* data; // (dim)^3 u8, coarser LOD chunk (NULL = none)
    int32_t        dim;  // side length of coarse chunk (64 for one level up)
} vl264_lod_ref;

// ── Output buffer ───────────────────────────────────────────────────────────

typedef struct vl264_buf {
    uint8_t* data;     // if NULL on encode, vl264 allocates (caller frees via vl264_free)
    size_t   size;     // bytes written after encode
    size_t   capacity; // allocated size
} vl264_buf;

void   vl264_free(void* ptr);
size_t vl264_max_compressed_size(void); // worst-case bound for pre-allocation

// ── Encoder ─────────────────────────────────────────────────────────────────

typedef struct vl264_enc vl264_enc;

vl264_enc* vl264_enc_create(const vl264_cfg* cfg);
void       vl264_enc_destroy(vl264_enc* e);

// Encode a full 128^3 chunk.
// chunk:     VL264_CHUNK_VOXELS u8, z-major order (data[z*128*128 + y*128 + x])
// neighbors: boundary faces from adjacent chunks (NULL = none)
// lod:       coarser LOD for delta coding (NULL = none)
// out:       output bitstream buffer
vl264_status vl264_encode(
    vl264_enc*             e,
    const uint8_t*         chunk,
    const vl264_neighbors* neighbors,
    const vl264_lod_ref*   lod,
    vl264_buf*             out);

// Comprehensive encode/decode statistics.
typedef struct vl264_stats {
    // ── Compression ─────────────────────────────────────────────────────
    vl264_axis axis;           // selected encode axis
    size_t     input_bytes;    // always VL264_CHUNK_VOXELS
    size_t     output_bytes;   // compressed size
    float      ratio;          // compression ratio (input / output)
    float      bits_per_voxel; // output_bytes * 8.0 / input_bytes

    // ── Timing (seconds) ────────────────────────────────────────────────
    double     encode_sec;     // wall-clock encode time
    double     decode_sec;     // wall-clock decode time
    double     encode_mbs;     // encode throughput (MB/s)
    double     decode_mbs;     // decode throughput (MB/s)

    // ── Quality — error distribution ────────────────────────────────────
    float      mse;            // mean squared error
    float      psnr;           // peak signal-to-noise ratio (dB)
    float      mae;            // mean absolute error
    float      max_err;        // maximum absolute error
    float      p50_err;        // 50th percentile (median) absolute error
    float      p90_err;        // 90th percentile absolute error
    float      p95_err;        // 95th percentile absolute error
    float      p99_err;        // 99th percentile absolute error

    // ── Coding statistics ───────────────────────────────────────────────
    float      avg_qp;         // average QP used across all blocks
    uint32_t   i_slices;       // number of I-frame slices
    uint32_t   p_slices;       // number of P-frame slices
    uint32_t   total_blocks;   // total 4x4 blocks encoded
    uint32_t   skip_blocks;    // blocks coded as skip (zero residual)
    uint32_t   intra_blocks;   // blocks coded as intra
    uint32_t   inter_blocks;   // blocks coded as inter (non-skip)
    uint32_t   zero_coeff_blocks; // blocks with all-zero quantized coefficients
    float      avg_nonzero_coeffs; // average non-zero coefficients per block

    // ── Input analysis ──────────────────────────────────────────────────
    float      input_mean;     // mean of input chunk
    float      input_stddev;   // standard deviation of input chunk
    uint8_t    input_min;      // minimum voxel value
    uint8_t    input_max;      // maximum voxel value
    float      input_entropy;  // Shannon entropy (bits per voxel)
} vl264_stats;

// Backward-compatible alias
typedef vl264_stats vl264_enc_stats;

vl264_status vl264_enc_stats_get(const vl264_enc* e, vl264_stats* s);

// Compute full quality stats by comparing original and decoded chunk.
// Fills the quality/error fields (mse, psnr, mae, max_err, percentiles).
// Also fills input analysis fields. Timing fields are NOT set by this call.
void vl264_stats_compute(vl264_stats* s,
                          const uint8_t* original,
                          const uint8_t* decoded,
                          size_t n);

// Print a human-readable stats report to a file (use stdout for console).
void vl264_stats_print(const vl264_stats* s, FILE* out);

// ── Decoder ─────────────────────────────────────────────────────────────────

typedef struct vl264_dec vl264_dec;

vl264_dec* vl264_dec_create(void);
void       vl264_dec_destroy(vl264_dec* d);

// Decode a full 128^3 chunk.
// bitstream:      encoded data from vl264_encode
// bitstream_size: byte length of encoded data
// neighbors/lod:  must match what was used during encode
// chunk_out:      VL264_CHUNK_VOXELS pre-allocated buffer
vl264_status vl264_decode(
    vl264_dec*             d,
    const uint8_t*         bitstream,
    size_t                 bitstream_size,
    const vl264_neighbors* neighbors,
    const vl264_lod_ref*   lod,
    uint8_t*               chunk_out);

// Streaming decode — decode one slice at a time.
vl264_status vl264_decode_begin(
    vl264_dec*             d,
    const uint8_t*         bitstream,
    size_t                 bitstream_size,
    const vl264_neighbors* neighbors,
    const vl264_lod_ref*   lod);

// Decode the next slice in coded order.
// slice_out:   128*128 bytes
// slice_index: receives the spatial slice index (0..127)
// Returns VL264_OK on success, VL264_ERR_OVERFLOW when all slices decoded.
vl264_status vl264_decode_next_slice(
    vl264_dec* d,
    uint8_t*   slice_out,
    uint32_t*  slice_index);

// Query the encode axis from a streaming decode session (call after decode_begin).
vl264_axis vl264_decode_axis(const vl264_dec* d);

// ── Utilities ───────────────────────────────────────────────────────────────

const char* vl264_version_str(void);

// Select best encode axis for a chunk (exposed for analysis/testing).
vl264_axis vl264_analyze_axis(const uint8_t* chunk);

// Compute PSNR between two buffers (dB). Returns INFINITY if identical.
float vl264_psnr(const uint8_t* a, const uint8_t* b, size_t n);

// Compute MSE between two buffers.
float vl264_mse(const uint8_t* a, const uint8_t* b, size_t n);

#endif // VL264_H
