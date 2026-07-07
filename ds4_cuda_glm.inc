/* =========================================================================
 * GLM 5.2 backend stubs.
 *
 * The GLM graph kernels are implemented only in the Metal backend so far;
 * ds4_engine_open() rejects GLM models on CUDA before any of these can
 * run. The stubs exist so the CUDA build links, and they fail loudly if
 * that invariant is ever broken.
 * ========================================================================= */

static void ds4_cuda_glm_stub_note(const char *fn) {
    static int printed;
    if (!printed) {
        printed = 1;
        fprintf(stderr, "ds4: BUG: %s reached on the CUDA backend, but GLM kernels are Metal-only\n", fn);
    }
}

/* =========================================================================
 * GLM router + routed-MoE CUDA implementation.
 *
 * Ports the CPU f32 reference numerics (ds4.c: sigmoid_stable/topk_desc
 * router, ds4_vec_dot_q{2,4,5,6}_K_f32 direct-dequant dots, silu) onto the
 * existing DS4 CUDA MoE kernel structure (one thread block per output row,
 * QK_K-block striding + shared-memory tree reduction). Everything below is
 * compiled as part of ds4_cuda.cu, so the statics/helpers of that file
 * (cuda_ok, cuda_model_range_ptr, dev_f16_to_f32, dev_q4_K_get_scale_min,
 * dev_q2_K_dot_f32, cuda_block_q2_K/q4_K, g_ssd_streaming_mode, ...) are in
 * scope.
 * ========================================================================= */

/* GGML tensor-type ids used by the GLM routed experts. */
#define GLM_CUDA_TENSOR_Q2_K 10u
#define GLM_CUDA_TENSOR_Q4_K 12u
#define GLM_CUDA_TENSOR_Q5_K 13u
#define GLM_CUDA_TENSOR_Q6_K 14u

/* Upper bound on n_expert_used the router kernel can select (GLM-5.2 uses 8;
 * DS4_MAX_EXPERT_USED is 8 as well — 32 leaves headroom). */
#define GLM_CUDA_MAX_TOPK 32u

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[CUDA_QK_K / 8];
    uint8_t  qs[CUDA_QK_K / 2];
} cuda_block_q5_K;

typedef struct {
    uint8_t  ql[CUDA_QK_K / 2];
    uint8_t  qh[CUDA_QK_K / 4];
    int8_t   scales[CUDA_QK_K / 16];
    uint16_t d;
} cuda_block_q6_K;

__device__ __forceinline__ static float glm_dev_sigmoid_stable(float x) {
    if (x >= 0.0f) {
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    } else {
        const float e = expf(x);
        return e / (1.0f + e);
    }
}

/* Direct dequant·f32 dots, faithful ports of ds4_vec_dot_q4_K_f32 /
 * ds4_vec_dot_q5_K_f32 / ds4_vec_dot_q6_K_f32 (ds4.c). dev_q2_K_dot_f32
 * already exists in ds4_cuda.cu with the same shape. */
__device__ static float dev_q4_K_dot_f32(const cuda_block_q4_K *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const cuda_block_q4_K *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const float dmin = dev_f16_to_f32(xb->dmin);
        const uint8_t *qs = xb->qs;
        const float *yb = x + (uint64_t)b * CUDA_QK_K;
        for (uint32_t j = 0; j < CUDA_QK_K / 32u; j++) {
            uint8_t sc_val, m_val;
            dev_q4_K_get_scale_min(j, xb->scales, &sc_val, &m_val);
            const uint32_t byte_off = (j >> 1) * 32u;
            const uint32_t shift = (j & 1u) * 4u;
            const float scale = d * (float)sc_val;
            const float minv = dmin * (float)m_val;
            for (uint32_t l = 0; l < 32u; l++) {
                const int q = (qs[byte_off + l] >> shift) & 0x0F;
                acc += (scale * (float)q - minv) * yb[j * 32u + l];
            }
        }
    }
    return acc;
}

__device__ static float dev_q5_K_dot_f32(const cuda_block_q5_K *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const cuda_block_q5_K *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const float dmin = dev_f16_to_f32(xb->dmin);
        const uint8_t *ql = xb->qs;
        const uint8_t *qh = xb->qh;
        const float *yb = x + (uint64_t)b * CUDA_QK_K;
        for (uint32_t group = 0; group < CUDA_QK_K / 32u; group++) {
            uint8_t sc_val, m_val;
            dev_q4_K_get_scale_min(group, xb->scales, &sc_val, &m_val);
            const uint32_t ql_base = (group >> 1) * 32u;
            const uint32_t shift = (group & 1u) * 4u;
            const uint8_t hmask = (uint8_t)(1u << group);
            const float scale = d * (float)sc_val;
            const float minv = dmin * (float)m_val;
            for (uint32_t l = 0; l < 32u; l++) {
                const int q = ((ql[ql_base + l] >> shift) & 0x0F) +
                              ((qh[l] & hmask) ? 16 : 0);
                acc += (scale * (float)q - minv) * yb[group * 32u + l];
            }
        }
    }
    return acc;
}

__device__ static float dev_q6_K_dot_f32(const cuda_block_q6_K *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const cuda_block_q6_K *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const uint8_t *ql = xb->ql;
        const uint8_t *qh = xb->qh;
        const int8_t *scales = xb->scales;
        const float *yb = x + (uint64_t)b * CUDA_QK_K;
        for (uint32_t n128 = 0; n128 < CUDA_QK_K; n128 += 128u) {
            for (uint32_t l = 0; l < 32u; l++) {
                const uint32_t is = l / 16u;
                const int q1 = ((int)(ql[l +  0u] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = ((int)(ql[l + 32u] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = ((int)(ql[l +  0u] >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = ((int)(ql[l + 32u] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32;
                acc += d * (float)scales[is + 0u] * (float)q1 * yb[n128 + l +  0u];
                acc += d * (float)scales[is + 2u] * (float)q2 * yb[n128 + l + 32u];
                acc += d * (float)scales[is + 4u] * (float)q3 * yb[n128 + l + 64u];
                acc += d * (float)scales[is + 6u] * (float)q4 * yb[n128 + l + 96u];
            }
            ql += 64;
            qh += 32;
            scales += 8;
        }
    }
    return acc;
}

/* Dot of one CUDA_QK_K-wide block `b` of a quantized weight row against the
 * matching f32 activation slice. `type` is uniform across a thread block. */
__device__ __forceinline__ static float glm_dev_expert_dot_block_f32(
        uint32_t type,
        const char *row_base,
        uint32_t b,
        const float *x) {
    const float *xb = x + (uint64_t)b * CUDA_QK_K;
    switch (type) {
    case GLM_CUDA_TENSOR_Q2_K:
        return dev_q2_K_dot_f32((const cuda_block_q2_K *)row_base + b, xb, 1);
    case GLM_CUDA_TENSOR_Q4_K:
        return dev_q4_K_dot_f32((const cuda_block_q4_K *)row_base + b, xb, 1);
    case GLM_CUDA_TENSOR_Q5_K:
        return dev_q5_K_dot_f32((const cuda_block_q5_K *)row_base + b, xb, 1);
    case GLM_CUDA_TENSOR_Q6_K:
        return dev_q6_K_dot_f32((const cuda_block_q6_K *)row_base + b, xb, 1);
    default:
        return 0.0f;
    }
}

/* One block per token. Threads cooperatively compute sigmoid probs into
 * shared memory (and the probs output); thread 0 then reproduces the CPU
 * topk_desc insertion sort (strict >, so ties keep the lower index) and the
 * unbiased-prob weight normalization. */
__global__ static void glm_router_select_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const float *logits,
        uint32_t n_expert,
        uint32_t n_expert_used,
        float expert_weight_scale,
        uint32_t n_tokens) {
    extern __shared__ float glm_sprob[];
    const uint32_t t = blockIdx.x;
    if (t >= n_tokens) return;
    const float *log = logits + (uint64_t)t * n_expert;
    float *prob = probs + (uint64_t)t * n_expert;
    int32_t *sel = selected + (uint64_t)t * n_expert_used;
    float *w = weights + (uint64_t)t * n_expert_used;

    for (uint32_t i = threadIdx.x; i < n_expert; i += blockDim.x) {
        const float p = glm_dev_sigmoid_stable(log[i]);
        glm_sprob[i] = p;
        prob[i] = p;
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    int32_t idx[GLM_CUDA_MAX_TOPK];
    for (uint32_t j = 0; j < n_expert_used; j++) idx[j] = -1;
    for (uint32_t i = 0; i < n_expert; i++) {
        const float score = glm_sprob[i] + bias[i];
        for (uint32_t j = 0; j < n_expert_used; j++) {
            if (idx[j] < 0 || score > glm_sprob[idx[j]] + bias[idx[j]]) {
                for (uint32_t m = n_expert_used - 1u; m > j; m--) idx[m] = idx[m - 1u];
                idx[j] = (int32_t)i;
                break;
            }
        }
    }

    float sum = 0.0f;
    for (uint32_t j = 0; j < n_expert_used; j++) {
        const int32_t e = idx[j];
        const float v = (e >= 0 && (uint32_t)e < n_expert) ? glm_sprob[e] : 0.0f;
        sel[j] = e;
        w[j] = v;
        sum += v;
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (uint32_t j = 0; j < n_expert_used; j++) {
        w[j] = w[j] / sum * expert_weight_scale;
    }
}

static int glm_router_select_launch(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t              n_expert,
        uint32_t              n_expert_used,
        float                 expert_weight_scale,
        uint32_t              n_tokens) {
    if (!selected || !weights || !probs || !logits || !model_map ||
        n_tokens == 0 || n_expert == 0 || n_expert_used == 0 ||
        n_expert_used > n_expert || n_expert_used > GLM_CUDA_MAX_TOPK ||
        logits->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        probs->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * n_expert_used * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert_used * sizeof(float)) {
        return 0;
    }
    const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
    if (bias_offset > model_size || model_size - bias_offset < bias_bytes) return 0;
    const float *bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, bias_bytes, "glm_router_bias");
    if (!bias) return 0;
    const uint32_t shared_bytes = (uint32_t)(n_expert * sizeof(float));
    glm_router_select_kernel<<<n_tokens, 256, shared_bytes>>>(
            (int32_t *)selected->ptr,
            (float *)weights->ptr,
            (float *)probs->ptr,
            bias,
            (const float *)logits->ptr,
            n_expert,
            n_expert_used,
            expert_weight_scale,
            n_tokens);
    return cuda_ok(cudaGetLastError(), "glm router_select launch");
}

/* One thread block per (token, slot, mid row): grid.x = mid row, grid.y =
 * token*n_expert + slot. Threads stride over the QK_K blocks of the row and
 * tree-reduce in shared memory; thread 0 applies silu(gate)*up*weight (no
 * clamp for GLM). Negative slots write 0, matching the Metal behavior. */
__global__ static void glm_moe_gate_up_mid_f32_kernel(
        float *mid,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes,
        uint32_t gate_type,
        uint32_t up_type,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t n_total_expert,
        uint64_t mid_token_stride) {
    const uint32_t row = blockIdx.x;
    const uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    const uint32_t tok = pair / n_expert;
    const uint32_t slot = pair - tok * n_expert;
    float *mid_out = mid + (uint64_t)tok * mid_token_stride + (uint64_t)slot * expert_mid_dim + row;
    const int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) {
        if (threadIdx.x == 0) *mid_out = 0.0f;
        return;
    }
    const uint32_t nb = expert_in_dim / CUDA_QK_K;
    const char *gr = gate_base + (uint64_t)(uint32_t)expert_i * gate_expert_bytes + (uint64_t)row * gate_row_bytes;
    const char *ur = up_base + (uint64_t)(uint32_t)expert_i * up_expert_bytes + (uint64_t)row * up_row_bytes;
    const float *xr = x + (uint64_t)tok * expert_in_dim;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        gate += glm_dev_expert_dot_block_f32(gate_type, gr, b, xr);
        up += glm_dev_expert_dot_block_f32(up_type, ur, b, xr);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        const float silu_gate = gate * glm_dev_sigmoid_stable(gate);
        *mid_out = silu_gate * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

/* Fused down projection: one thread block per (token, out row); threads
 * stride over slot×QK_K-block pairs so all selected experts of the token
 * accumulate into one output value. Writes out (no accumulation) — the
 * shared expert is added by the caller. */
__global__ static void glm_moe_down_out_f32_kernel(
        float *out,
        const char *down_base,
        const float *mid,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t down_type,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t n_total_expert,
        uint64_t mid_token_stride) {
    const uint32_t row = blockIdx.x;
    const uint32_t tok = blockIdx.y;
    if (row >= out_dim) return;
    const uint32_t nb = expert_mid_dim / CUDA_QK_K;
    const uint32_t total = n_expert * nb;
    float acc = 0.0f;
    for (uint32_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const uint32_t slot = idx / nb;
        const uint32_t b = idx - slot * nb;
        const int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
        if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) continue;
        const char *dr = down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes;
        const float *xr = mid + (uint64_t)tok * mid_token_stride + (uint64_t)slot * expert_mid_dim;
        acc += glm_dev_expert_dot_block_f32(down_type, dr, b, xr);
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[(uint64_t)tok * out_dim + row] = partial[0];
}

static bool glm_cuda_gate_pair_type_supported(uint32_t gate_type, uint32_t up_type) {
    return gate_type == up_type &&
           (gate_type == GLM_CUDA_TENSOR_Q2_K ||
            gate_type == GLM_CUDA_TENSOR_Q4_K ||
            gate_type == GLM_CUDA_TENSOR_Q5_K);
}

static bool glm_cuda_down_type_supported(uint32_t down_type) {
    return down_type == GLM_CUDA_TENSOR_Q2_K ||
           down_type == GLM_CUDA_TENSOR_Q4_K ||
           down_type == GLM_CUDA_TENSOR_Q5_K ||
           down_type == GLM_CUDA_TENSOR_Q6_K;
}

/* Dense K-quant row bytes for expert_in-wide rows; 0 when unsupported. */
static uint64_t glm_cuda_row_bytes_for_type(uint32_t type, uint32_t in_dim) {
    const uint64_t nb = in_dim / CUDA_QK_K;
    switch (type) {
    case GLM_CUDA_TENSOR_Q2_K: return nb * sizeof(cuda_block_q2_K);
    case GLM_CUDA_TENSOR_Q4_K: return nb * sizeof(cuda_block_q4_K);
    case GLM_CUDA_TENSOR_Q5_K: return nb * sizeof(cuda_block_q5_K);
    case GLM_CUDA_TENSOR_Q6_K: return nb * sizeof(cuda_block_q6_K);
    default: return 0;
    }
}

static int glm_routed_moe_launch(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              gate_type,
        uint32_t              up_type,
        uint32_t              down_type,
        uint64_t              gate_expert_bytes,
        uint64_t              gate_row_bytes,
        uint64_t              up_expert_bytes,
        uint64_t              up_row_bytes,
        uint64_t              down_expert_bytes,
        uint64_t              down_row_bytes,
        uint32_t              expert_in_dim,
        uint32_t              expert_mid_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t              n_total_expert,
        uint32_t              n_expert,
        uint32_t              layer_index,
        const ds4_gpu_tensor *x,
        uint32_t              n_tokens,
        uint64_t              mid_token_stride) {
    (void)layer_index;
    if (!out || !mid || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_total_expert == 0 || n_expert == 0 ||
        expert_in_dim == 0 || expert_mid_dim == 0 || out_dim == 0 ||
        expert_in_dim % CUDA_QK_K != 0 || expert_mid_dim % CUDA_QK_K != 0 ||
        !glm_cuda_gate_pair_type_supported(gate_type, up_type) ||
        !glm_cuda_down_type_supported(down_type)) {
        return 0;
    }
    if (gate_row_bytes != glm_cuda_row_bytes_for_type(gate_type, expert_in_dim) ||
        up_row_bytes != glm_cuda_row_bytes_for_type(up_type, expert_in_dim) ||
        down_row_bytes != glm_cuda_row_bytes_for_type(down_type, expert_mid_dim) ||
        gate_expert_bytes != (uint64_t)expert_mid_dim * gate_row_bytes ||
        up_expert_bytes != (uint64_t)expert_mid_dim * up_row_bytes ||
        down_expert_bytes != (uint64_t)out_dim * down_row_bytes) {
        return 0;
    }
    if (mid_token_stride < (uint64_t)n_expert * expert_mid_dim) return 0;
    if (x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * n_expert * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        mid->bytes < ((uint64_t)(n_tokens - 1u) * mid_token_stride +
                      (uint64_t)n_expert * expert_mid_dim) * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }
    const uint64_t gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t up_bytes = (uint64_t)n_total_expert * up_expert_bytes;
    const uint64_t down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_offset > model_size || gate_bytes > model_size - gate_offset ||
        up_offset > model_size || up_bytes > model_size - up_offset ||
        down_offset > model_size || down_bytes > model_size - down_offset) {
        return 0;
    }
    /* grid.y limits: pairs for the mid kernel, tokens for the down kernel. */
    if ((uint64_t)n_tokens * n_expert > 65535u) return 0;

    /* MVP weight addressing: read straight from the (cached/registered)
     * model mapping in all modes. The streaming expert cache is only a
     * prefetch hint here — see the *_begin_selected_load_tensor wrapper. */
    const char *gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_bytes, "glm_moe_gate");
    const char *up_w = cuda_model_range_ptr(model_map, up_offset, up_bytes, "glm_moe_up");
    const char *down_w = cuda_model_range_ptr(model_map, down_offset, down_bytes, "glm_moe_down");
    if (!gate_w || !up_w || !down_w) return 0;

    dim3 mgrid(expert_mid_dim, n_tokens * n_expert, 1);
    glm_moe_gate_up_mid_f32_kernel<<<mgrid, 256>>>(
            (float *)mid->ptr,
            gate_w,
            up_w,
            (const float *)x->ptr,
            (const int32_t *)selected->ptr,
            (const float *)weights->ptr,
            gate_expert_bytes,
            gate_row_bytes,
            up_expert_bytes,
            up_row_bytes,
            gate_type,
            up_type,
            expert_in_dim,
            expert_mid_dim,
            n_expert,
            n_total_expert,
            mid_token_stride);
    int ok = cuda_ok(cudaGetLastError(), "glm routed_moe gate/up launch");
    if (ok) {
        dim3 dgrid(out_dim, n_tokens, 1);
        glm_moe_down_out_f32_kernel<<<dgrid, 256>>>(
                (float *)out->ptr,
                down_w,
                (const float *)mid->ptr,
                (const int32_t *)selected->ptr,
                down_expert_bytes,
                down_row_bytes,
                down_type,
                expert_mid_dim,
                out_dim,
                n_expert,
                n_total_expert,
                mid_token_stride);
        ok = cuda_ok(cudaGetLastError(), "glm routed_moe down launch");
    }
    return ok;
}

/* =========================================================================
 * GLM elementwise / embed / swiglu / rope kernels.
 *
 * Straight ports of the Metal GLM kernels (metal/dsv4_misc.metal,
 * metal/get_rows.metal) onto the existing DS4 CUDA kernel styles
 * (add_kernel, rms_norm_weight_kernel, rope_tail_kernel, ...). Numerics
 * match the CPU references in ds4.c (rms_norm_weight, swiglu/silu,
 * rope_tail_ext_inplace).
 * ========================================================================= */

/* out[i] = a[i] + b[i] + c[i] — clone of add_kernel with a third input. */
__global__ static void glm_add3_kernel(
        float *out,
        const float *a,
        const float *b,
        const float *c,
        uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i] + c[i];
}

/* Fused residual add + RMS norm with learned weight. Single row:
 * s[i] = a[i] + b[i]; sum_out[i] = s[i];
 * scale = 1/sqrt(mean(s^2) + eps); norm_out[i] = s[i]*scale*w[i].
 * Clone of rms_norm_weight_kernel with the extra add + sum_out store. */
__global__ static void glm_add_rms_norm_weight_kernel(
        float *norm_out,
        float *sum_out,
        const float *a,
        const float *b,
        const float *w,
        uint32_t n,
        float eps) {
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = a[i] + b[i];
        sum_out[i] = v;
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        norm_out[i] = sum_out[i] * scale * w[i];
    }
}

/* Batched Q8_0 embedding gather. One thread per (token, dim) element;
 * invalid tokens skip the write (matching kernel_get_rows_q8_0_f32). */
__global__ static void glm_embed_tokens_q8_0_kernel(
        float *out,
        const int32_t *tokens,
        const unsigned char *w,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint64_t row_blocks) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * n_embd;
    if (gid >= n) return;
    const uint32_t d = (uint32_t)(gid % n_embd);
    const uint32_t t = (uint32_t)(gid / n_embd);
    const int32_t tok = tokens[t];
    if (tok < 0 || (uint32_t)tok >= n_vocab) return;
    const uint64_t b = d / 32u;
    const uint32_t j = d - (uint32_t)b * 32u;
    const unsigned char *blk = w + ((uint64_t)(uint32_t)tok * row_blocks + b) * 34u;
    const float scale = __half2float(*(const __half *)blk);
    const int8_t q = *(const int8_t *)(blk + 2 + j);
    out[gid] = scale * (float)q;
}

/* Per-token-row RMS norm over the first kv_lora_dim of each kv_raw_dim-wide
 * input row (port of kernel_glm_kv_lora_rms_norm; CPU ref rms_norm_weight,
 * float accumulation on GPU). One block per token. */
__global__ static void glm_kv_lora_rms_norm_kernel(
        float *out,
        const float *kv_raw,
        const float *w,
        uint32_t n_tokens,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= n_tokens) return;
    const float *x = kv_raw + (uint64_t)row * kv_raw_dim;
    float *o = out + (uint64_t)row * kv_lora_dim;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < kv_lora_dim; i += blockDim.x) {
        const float v = x[i];
        ss += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / (float)kv_lora_dim + eps);
    for (uint32_t i = threadIdx.x; i < kv_lora_dim; i += blockDim.x) {
        o[i] = x[i] * inv * w[i];
    }
}

/* Generalized rope_tail_kernel (ds4_cuda.cu): the rotated slice starts at
 * rot_offset inside each head instead of the hardcoded head_dim - n_rot
 * tail. Pairs are consecutive (i, i+1); yarn math identical to the CPU
 * reference rope_tail_ext_inplace (ds4.c). */
__global__ static void glm_rope_offset_kernel(
        float *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t rot_offset,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pairs = n_tok * n_head * (n_rot / 2);
    if (gid >= pairs) return;
    uint32_t pair = gid % (n_rot / 2);
    uint32_t tmp = gid / (n_rot / 2);
    uint32_t h = tmp % n_head;
    uint32_t t = tmp / n_head;
    uint32_t i = pair * 2;

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }

    float theta_extrap = (float)(pos0 + t) * powf(freq_base, -((float)i) / (float)n_rot);
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    float c = cosf(theta) * mscale;
    float s = sinf(theta) * mscale;

    float *rot = x + ((uint64_t)t * n_head + h) * head_dim + rot_offset;
    float x0 = rot[i];
    float x1 = rot[i + 1];
    rot[i] = x0 * c - x1 * s;
    rot[i + 1] = x0 * s + x1 * c;
}

__global__ static void glm_fill_selected_range_kernel(
        uint32_t *selected,
        uint32_t n_selected) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n_selected) selected[gid] = gid;
}

/* selected[token*n_selected + slot] = slot < pos0+token+1 ? slot : pad_row
 * (port of kernel_glm_fill_selected_range_batch). */
__global__ static void glm_fill_selected_range_batch_kernel(
        uint32_t *selected,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_selected,
        uint32_t pad_row) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = n_tokens * n_selected;
    if (gid >= total) return;
    const uint32_t token = gid / n_selected;
    const uint32_t slot = gid - token * n_selected;
    const uint32_t visible = pos0 + token + 1u;
    selected[gid] = slot < visible ? slot : pad_row;
}

/* Dedicated device scratch for the mid-only shared swiglu. Kept separate
 * from cuda_tmp_alloc's arena because the q8_0 pair matmul we call uses
 * that arena for its own activation prequant. */
static void *g_glm_swiglu_scratch = NULL;
static uint64_t g_glm_swiglu_scratch_bytes = 0;

static void *glm_swiglu_scratch_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_glm_swiglu_scratch_bytes >= bytes) return g_glm_swiglu_scratch;
    if (g_glm_swiglu_scratch) {
        (void)cudaFree(g_glm_swiglu_scratch);
        g_glm_swiglu_scratch = NULL;
        g_glm_swiglu_scratch_bytes = 0;
    }
    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA GLM scratch alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "scratch", (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    g_glm_swiglu_scratch = ptr;
    g_glm_swiglu_scratch_bytes = bytes;
    return g_glm_swiglu_scratch;
}

/* Shared launcher for the two GLM rope entry points: rotates rot_dim
 * consecutive-pair values starting at rot_offset inside each head. */
static int glm_rope_offset_launch(
        ds4_gpu_tensor *x,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        rot_dim,
        uint32_t        rot_offset,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow,
        const char     *label) {
    if (!x || n_tokens == 0 || n_head == 0 || head_dim == 0 ||
        rot_dim == 0 || (rot_dim & 1u) != 0 ||
        rot_offset > head_dim || rot_dim > head_dim - rot_offset ||
        x->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float)) {
        return 0;
    }
    const uint32_t pairs = n_tokens * n_head * (rot_dim / 2);
    glm_rope_offset_kernel<<<(pairs + 255) / 256, 256>>>(
            (float *)x->ptr,
            n_tokens,
            n_head,
            head_dim,
            rot_dim,
            rot_offset,
            pos0,
            n_ctx_orig,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow);
    return cuda_ok(cudaGetLastError(), label);
}

/* =========================================================================
 * GLM attention / compact-KV / indexer core kernels.
 *
 * Ports of the Metal shaders in metal/dsv4_misc.metal (kernel_glm_store_*,
 * kernel_glm_k_b_project_q8_0, kernel_glm_qk_lowrank_q8_0,
 * kernel_glm_value_project_q8_0_batch_heads, kernel_glm_build_kv_cache,
 * kernel_glm_attention_full, kernel_glm_attention_indexed_decode,
 * kernel_glm_indexer_score_one, kernel_glm_indexer_scores_batch) with
 * numerics matching the CPU references layer_glm_attention_prefill_f32_ref
 * and glm_k_b_project_f32_ref (ds4.c). Caches are f16 or f32 selected by
 * the cache_f16 flag (compile-time template parameter here).
 * ========================================================================= */

/* f16/f32 cache element access, compile-time selected. */
template <bool CF16>
__device__ __forceinline__ static float glm_dev_cache_load(
        const char *base, uint64_t index) {
    if (CF16) return __half2float(((const __half *)base)[index]);
    return ((const float *)base)[index];
}

template <bool CF16>
__device__ __forceinline__ static void glm_dev_cache_store(
        char *base, uint64_t index, float x) {
    if (CF16) ((__half *)base)[index] = __float2half(x);
    else ((float *)base)[index] = x;
}

/* Yarn correction dims, same math as glm_rope_offset_kernel (and the CPU
 * reference rope_tail_ext_inplace). No-op when ext_factor == 0 (all GLM
 * call sites). */
__device__ __forceinline__ static void glm_dev_rope_corr(
        uint32_t rot_dim,
        uint32_t n_ctx_orig,
        float freq_base,
        float beta_fast,
        float beta_slow,
        float ext_factor,
        float *corr0,
        float *corr1) {
    *corr0 = 0.0f;
    *corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        const float denom = 2.0f * logf(freq_base);
        float c0 = floorf((float)rot_dim * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        float c1 = ceilf((float)rot_dim * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        *corr0 = fmaxf(0.0f, c0);
        *corr1 = fminf((float)(rot_dim - 1), c1);
    }
}

/* cos/sin for the adjacent pair starting at even index r, rotating at
 * position pos. theta = pos * freq_base^(-r/rot_dim) with the same yarn
 * blend as glm_rope_offset_kernel. */
__device__ __forceinline__ static void glm_dev_rope_cs(
        float pos,
        uint32_t r,
        uint32_t rot_dim,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float corr0,
        float corr1,
        float *c,
        float *s) {
    const float theta_extrap = pos * powf(freq_base, -((float)r) / (float)rot_dim);
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        const float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)r) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *c = cosf(theta) * mscale;
    *s = sinf(theta) * mscale;
}

/* Dot of one Q8_0 row (34-byte blocks: f16 scale + 32 int8) against n f32
 * activations. Same accumulation order as matvec_q8_0_f32_ref (ds4.c). */
__device__ static float glm_dev_q8_row_dot_f32(
        const char *row, const float *x, uint32_t n) {
    float acc = 0.0f;
    const uint32_t blocks = (n + 31u) / 32u;
    for (uint32_t b = 0; b < blocks; b++) {
        const char *block_base = row + (uint64_t)b * 34u;
        const float d = dev_f16_to_f32(*(const uint16_t *)block_base);
        const int8_t *qs = (const int8_t *)(block_base + 2);
        const uint32_t i0 = b * 32u;
        const uint32_t nb = n - i0 < 32u ? n - i0 : 32u;
        float bsum = 0.0f;
        for (uint32_t i = 0; i < nb; i++) {
            bsum += (float)qs[i] * x[i0 + i];
        }
        acc += d * bsum;
    }
    return acc;
}

static uint64_t glm_cuda_q8_row_bytes(uint32_t n) {
    return (((uint64_t)n + 31u) / 32u) * 34u;
}

/* Opt in to >48KB dynamic shared memory when the device allows it (same
 * pattern as the indexer_topk_8192_cub launch in ds4_cuda.cu). Returns 1
 * when the requested size is usable. */
static int glm_cuda_ensure_dyn_shmem(const void *kernel_func, uint64_t shmem) {
    if (shmem <= 48u * 1024u) return 1;
    int dev = 0;
    int max_optin = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err == cudaSuccess) {
        err = cudaDeviceGetAttribute(&max_optin,
                                     cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                     dev);
    }
    if (err != cudaSuccess || (uint64_t)max_optin < shmem) return 0;
    err = cudaFuncSetAttribute(kernel_func,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               (int)shmem);
    return err == cudaSuccess;
}

/* --- A: store indexer K row (centered layer norm + rope on first rot_dim,
 *        port of kernel_glm_store_indexer_k). One block per token. --- */
template <bool CF16>
__global__ static void glm_store_indexer_k_kernel(
        char *indexer_key_cache,
        const float *raw_k,
        const float *w,
        const float *b,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_cap,
        uint32_t head_dim,
        uint32_t rot_dim,
        uint32_t n_ctx_orig,
        float eps,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint32_t token = blockIdx.x;
    if (token >= n_tokens) return;
    const uint32_t pos = pos0 + token;
    if (pos >= cache_cap) return;

    const uint32_t nth = blockDim.x;
    const float *src = raw_k + (uint64_t)token * head_dim;

    __shared__ float red[128];
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += nth) sum += src[i];
    red[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = nth >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    const float mean = red[0] / (float)head_dim;
    __syncthreads();

    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += nth) {
        const float d = src[i] - mean;
        ss += d * d;
    }
    red[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t stride = nth >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)head_dim + eps);

    float corr0, corr1;
    glm_dev_rope_corr(rot_dim, n_ctx_orig, freq_base, beta_fast, beta_slow,
                      ext_factor, &corr0, &corr1);

    char *dst = indexer_key_cache +
        (uint64_t)pos * head_dim * (CF16 ? sizeof(__half) : sizeof(float));
    for (uint32_t i = threadIdx.x; i < head_dim; i += nth) {
        if (i < rot_dim) {
            if ((i & 1u) != 0u) continue;
            float c, s;
            glm_dev_rope_cs((float)pos, i, rot_dim,
                            freq_base, freq_scale, ext_factor, attn_factor,
                            corr0, corr1, &c, &s);
            const uint32_t j = i + 1u;
            const float x0 = (src[i] - mean) * inv * w[i] + b[i];
            const float x1 = (src[j] - mean) * inv * w[j] + b[j];
            glm_dev_cache_store<CF16>(dst, i, x0 * c - x1 * s);
            glm_dev_cache_store<CF16>(dst, j, x0 * s + x1 * c);
        } else {
            const float x = (src[i] - mean) * inv * w[i] + b[i];
            glm_dev_cache_store<CF16>(dst, i, x);
        }
    }
}

/* --- B: compact (DSA) cache store — kv_lora row from kv_norm plus RAW
 *        (unrotated) rope tail from kv_raw. One thread per element. --- */
template <bool CF16>
__global__ static void glm_store_compact_kv_kernel(
        char *kv_lora_cache,
        char *k_rope_cache,
        const float *kv_norm,
        const float *kv_raw,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_cap,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        uint32_t qk_rope) {
    const uint32_t per_tok = kv_lora_dim + qk_rope;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_tokens * per_tok;
    if (gid >= total) return;
    const uint32_t token = (uint32_t)(gid / per_tok);
    const uint32_t i = (uint32_t)(gid - (uint64_t)token * per_tok);
    const uint32_t pos = pos0 + token;
    if (pos >= cache_cap) return;
    if (i < kv_lora_dim) {
        const float v = kv_norm[(uint64_t)token * kv_lora_dim + i];
        glm_dev_cache_store<CF16>(kv_lora_cache, (uint64_t)pos * kv_lora_dim + i, v);
    } else {
        const uint32_t r = i - kv_lora_dim;
        const float v = kv_raw[(uint64_t)token * kv_raw_dim + kv_lora_dim + r];
        glm_dev_cache_store<CF16>(k_rope_cache, (uint64_t)pos * qk_rope + r, v);
    }
}

/* --- C part 1: RMS norm the first kv_lora_dim of kv_raw straight into the
 *        compact kv_lora cache row (f16/f32). One block per token. --- */
template <bool CF16>
__global__ static void glm_kv_norm_store_cache_kernel(
        char *kv_lora_cache,
        const float *kv_raw,
        const float *w,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_cap,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        float eps) {
    const uint32_t token = blockIdx.x;
    if (token >= n_tokens) return;
    const uint32_t pos = pos0 + token;
    if (pos >= cache_cap) return;
    const float *x = kv_raw + (uint64_t)token * kv_raw_dim;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < kv_lora_dim; i += blockDim.x) {
        const float v = x[i];
        ss += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / (float)kv_lora_dim + eps);
    char *dst = kv_lora_cache +
        (uint64_t)pos * kv_lora_dim * (CF16 ? sizeof(__half) : sizeof(float));
    for (uint32_t i = threadIdx.x; i < kv_lora_dim; i += blockDim.x) {
        glm_dev_cache_store<CF16>(dst, i, x[i] * inv * w[i]);
    }
}

/* --- C part 2: copy the raw rope tail into k_rope_cache. --- */
template <bool CF16>
__global__ static void glm_store_rope_tail_kernel(
        char *k_rope_cache,
        const float *kv_raw,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_cap,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        uint32_t qk_rope) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_tokens * qk_rope;
    if (gid >= total) return;
    const uint32_t token = (uint32_t)(gid / qk_rope);
    const uint32_t r = (uint32_t)(gid - (uint64_t)token * qk_rope);
    const uint32_t pos = pos0 + token;
    if (pos >= cache_cap) return;
    const float v = kv_raw[(uint64_t)token * kv_raw_dim + kv_lora_dim + r];
    glm_dev_cache_store<CF16>(k_rope_cache, (uint64_t)pos * qk_rope + r, v);
}

/* --- D: k_b projection. out[t,head,q] = sum_j deq(k_b[head,j])[q] *
 *        kv_norm[t,j]. Weight row for (head, j) is at
 *        (head*kv_lora_dim + j)*row_bytes and holds qk_nope values —
 *        layout confirmed by glm_k_b_project_f32_ref (ds4.c:11759).
 *        One block per (token, head); one thread per output q; the
 *        kv_norm row is staged in shared memory. --- */
__global__ static void glm_k_b_project_q8_0_kernel(
        float *out,
        const char *weight,
        const float *kv_norm,
        uint32_t n_tokens,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t n_head,
        uint64_t row_bytes) {
    extern __shared__ float glm_kbp_sh[];
    const uint32_t token = blockIdx.x;
    const uint32_t head = blockIdx.y;
    if (token >= n_tokens || head >= n_head) return;
    const float *kv = kv_norm + (uint64_t)token * kv_lora_dim;
    for (uint32_t j = threadIdx.x; j < kv_lora_dim; j += blockDim.x) {
        glm_kbp_sh[j] = kv[j];
    }
    __syncthreads();
    const uint32_t q = threadIdx.x;
    if (q >= qk_nope) return;
    const uint32_t block = q >> 5;
    const uint32_t qi = q & 31u;
    const char *wh = weight + (uint64_t)head * kv_lora_dim * row_bytes;
    float acc = 0.0f;
    for (uint32_t j = 0; j < kv_lora_dim; j++) {
        const char *block_base = wh + (uint64_t)j * row_bytes + (uint64_t)block * 34u;
        const float d = dev_f16_to_f32(*(const uint16_t *)block_base);
        const int8_t *qs = (const int8_t *)(block_base + 2);
        acc += d * (float)qs[qi] * glm_kbp_sh[j];
    }
    out[((uint64_t)token * n_head + head) * qk_nope + q] = acc;
}

/* --- E: transpose contraction of D — qk_low[head,j] =
 *        dot(deq(k_b[head,j])[0..qk_nope), q[head][0..qk_nope)). Same
 *        weight rows as D, contracted along the row instead. One block
 *        per head; the first qk_nope q values staged in shared. --- */
__global__ static void glm_qk_lowrank_q8_0_kernel(
        float *qk_low,
        const char *weight,
        const float *q,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_dim,
        uint64_t row_bytes) {
    extern __shared__ float glm_qkl_sh[];
    const uint32_t head = blockIdx.x;
    if (head >= n_head) return;
    const float *qh = q + (uint64_t)head * qk_dim;
    for (uint32_t d = threadIdx.x; d < qk_nope; d += blockDim.x) {
        glm_qkl_sh[d] = qh[d];
    }
    __syncthreads();
    float *out = qk_low + (uint64_t)head * kv_lora_dim;
    for (uint32_t j = threadIdx.x; j < kv_lora_dim; j += blockDim.x) {
        const char *row = weight + ((uint64_t)head * kv_lora_dim + j) * row_bytes;
        out[j] = glm_dev_q8_row_dot_f32(row, glm_qkl_sh, qk_nope);
    }
}

/* --- F: value projection. heads[t,head,d] = sum_j deq(v_b[head,d])[j] *
 *        lora[t,head,j]; v_b row for (head, d) at (head*value_dim + d) *
 *        row_bytes over kv_lora_dim inputs. Grid (head, token); the lora
 *        row is staged in shared memory. --- */
__global__ static void glm_value_project_q8_0_batch_heads_kernel(
        float *heads,
        const char *weight,
        const float *lora,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t value_dim,
        uint64_t row_bytes) {
    extern __shared__ float glm_vp_sh[];
    const uint32_t head = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (head >= n_head || token >= n_tokens) return;
    const float *src = lora + ((uint64_t)token * n_head + head) * kv_lora_dim;
    for (uint32_t j = threadIdx.x; j < kv_lora_dim; j += blockDim.x) {
        glm_vp_sh[j] = src[j];
    }
    __syncthreads();
    float *out = heads + ((uint64_t)token * n_head + head) * value_dim;
    for (uint32_t d = threadIdx.x; d < value_dim; d += blockDim.x) {
        const char *row = weight + ((uint64_t)head * value_dim + d) * row_bytes;
        out[d] = glm_dev_q8_row_dot_f32(row, glm_vp_sh, kv_lora_dim);
    }
}

/* --- G: expand the dense caches: key = k_nope ++ rotated rope tail (shared
 *        across heads, read from kv_raw), value copied. Port of
 *        kernel_glm_build_kv_cache; CPU ref ds4.c:11891-11899. --- */
template <bool CF16>
__global__ static void glm_build_kv_cache_kernel(
        char *key_cache,
        char *value_cache,
        const float *kv_raw,
        const float *k_nope,
        const float *value,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_cap,
        uint32_t n_head,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope,
        uint32_t value_dim,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint32_t token = blockIdx.x;
    const uint32_t head = blockIdx.y;
    if (token >= n_tokens || head >= n_head) return;
    const uint32_t pos = pos0 + token;
    if (pos >= cache_cap) return;

    const uint32_t nth = blockDim.x;
    const uint32_t qk_dim = qk_nope + qk_rope;
    const float *raw = kv_raw + (uint64_t)token * kv_raw_dim;
    const float *kn = k_nope + ((uint64_t)token * n_head + head) * qk_nope;
    const float *val = value + ((uint64_t)token * n_head + head) * value_dim;
    const uint64_t kbase = ((uint64_t)pos * n_head + head) * qk_dim;
    const uint64_t vbase = ((uint64_t)pos * n_head + head) * value_dim;

    for (uint32_t i = threadIdx.x; i < qk_nope; i += nth) {
        glm_dev_cache_store<CF16>(key_cache, kbase + i, kn[i]);
    }

    float corr0, corr1;
    glm_dev_rope_corr(qk_rope, n_ctx_orig, freq_base, beta_fast, beta_slow,
                      ext_factor, &corr0, &corr1);
    for (uint32_t r = threadIdx.x * 2u; r < qk_rope; r += nth * 2u) {
        float c, s;
        glm_dev_rope_cs((float)pos, r, qk_rope,
                        freq_base, freq_scale, ext_factor, attn_factor,
                        corr0, corr1, &c, &s);
        const float x0 = raw[kv_lora_dim + r];
        const float x1 = raw[kv_lora_dim + r + 1u];
        glm_dev_cache_store<CF16>(key_cache, kbase + qk_nope + r, x0 * c - x1 * s);
        glm_dev_cache_store<CF16>(key_cache, kbase + qk_nope + r + 1u, x0 * s + x1 * c);
    }

    for (uint32_t i = threadIdx.x; i < value_dim; i += nth) {
        glm_dev_cache_store<CF16>(value_cache, vbase + i, val[i]);
    }
}

/* --- H: dense causal attention on the expanded caches. Single-pass online
 *        softmax per (token, head) block: running max m / running sum l,
 *        chunk of 256 scores in shared, per-thread value accumulator in
 *        registers. Numerics match layer_glm_attention_prefill_f32_ref
 *        (ds4.c:11902-11934): scale*dot, max-subtract softmax, denom
 *        clamped at 1e-20. --- */
#define GLM_CUDA_ATTN_QK_MAX 512u
#define GLM_CUDA_ATTN_VACC_MAX 4u

template <bool CF16>
__global__ static void glm_attention_full_kernel(
        float *heads,
        const float *q,
        const char *key_cache,
        const char *value_cache,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_len,
        uint32_t n_head,
        uint32_t qk_dim,
        uint32_t value_dim,
        float scale) {
    const uint32_t token = blockIdx.x;
    const uint32_t head = blockIdx.y;
    if (token >= n_tokens || head >= n_head) return;
    const uint32_t nth = blockDim.x;
    const uint32_t visible = min(cache_len, pos0 + token + 1u);
    if (visible == 0) return;

    __shared__ float q_sh[GLM_CUDA_ATTN_QK_MAX];
    __shared__ float p_sh[256];
    __shared__ float red[256];

    const float *qh = q + ((uint64_t)token * n_head + head) * qk_dim;
    for (uint32_t i = threadIdx.x; i < qk_dim; i += nth) q_sh[i] = qh[i];
    __syncthreads();

    const uint32_t n_own = (value_dim + nth - 1u) / nth;
    float vacc[GLM_CUDA_ATTN_VACC_MAX];
    for (uint32_t k = 0; k < GLM_CUDA_ATTN_VACC_MAX; k++) vacc[k] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    for (uint32_t s0 = 0; s0 < visible; s0 += nth) {
        const uint32_t s = s0 + threadIdx.x;
        float score = -INFINITY;
        if (s < visible) {
            const uint64_t kbase = ((uint64_t)s * n_head + head) * qk_dim;
            float dotv = 0.0f;
            for (uint32_t i = 0; i < qk_dim; i++) {
                dotv += q_sh[i] * glm_dev_cache_load<CF16>(key_cache, kbase + i);
            }
            score = dotv * scale;
        }
        red[threadIdx.x] = score;
        __syncthreads();
        for (uint32_t stride = nth >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
            __syncthreads();
        }
        const float chunk_max = red[0];
        __syncthreads();
        const float m_new = fmaxf(m, chunk_max);

        const float p = (s < visible) ? expf(score - m_new) : 0.0f;
        p_sh[threadIdx.x] = p;
        red[threadIdx.x] = p;
        __syncthreads();
        for (uint32_t stride = nth >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
            __syncthreads();
        }
        const float chunk_sum = red[0];

        const float rescale = expf(m - m_new);
        l = l * rescale + chunk_sum;
        for (uint32_t k = 0; k < n_own; k++) vacc[k] *= rescale;

        const uint32_t chunk_len = min(nth, visible - s0);
        for (uint32_t r = 0; r < chunk_len; r++) {
            const float wgt = p_sh[r];
            if (wgt == 0.0f) continue;
            const uint64_t vbase = ((uint64_t)(s0 + r) * n_head + head) * value_dim;
            for (uint32_t k = 0; k < n_own; k++) {
                const uint32_t d = threadIdx.x + k * nth;
                if (d < value_dim) {
                    vacc[k] += wgt * glm_dev_cache_load<CF16>(value_cache, vbase + d);
                }
            }
        }
        m = m_new;
        __syncthreads();
    }

    const float denom = fmaxf(l, 1.0e-20f);
    float *out = heads + ((uint64_t)token * n_head + head) * value_dim;
    for (uint32_t k = 0; k < n_own; k++) {
        const uint32_t d = threadIdx.x + k * nth;
        if (d < value_dim) out[d] = vacc[k] / denom;
    }
}

/* --- I: indexed decode attention on the compact cache (single token).
 *        Port of kernel_glm_attention_indexed_decode: per selected row,
 *        score = qk_low·kv_lora_row + q_rope·rotate(k_rope_row) (rope
 *        position = absolute cache row index), softmax over n_selected,
 *        probability-weighted lora_sum, then the Q8_0 v_b projection.
 *        One block per head; scores live in dynamic shared memory. --- */
template <bool CF16>
__global__ static void glm_attention_indexed_decode_kernel(
        float *heads,
        const float *q,
        const float *qk_low,
        const char *kv_lora_cache,
        const char *k_rope_cache,
        const char *value_weight,
        const uint32_t *selected,
        uint32_t n_selected,
        uint32_t cache_cap,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope,
        uint32_t value_dim,
        uint64_t value_row_bytes,
        uint32_t n_ctx_orig,
        float scale,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    /* Dynamic shared layout:
     * scores[n_selected] | lora_sum[kv_lora_dim] | low_sh[kv_lora_dim] |
     * q_rope_sh[qk_rope] | red[256] */
    extern __shared__ float glm_aid_sh[];
    float *scores = glm_aid_sh;
    float *lora_sum = scores + n_selected;
    float *low_sh = lora_sum + kv_lora_dim;
    float *q_rope_sh = low_sh + kv_lora_dim;
    float *red = q_rope_sh + qk_rope;

    const uint32_t head = blockIdx.x;
    if (head >= n_head || n_selected == 0u) return;
    const uint32_t nth = blockDim.x;
    const uint32_t qk_dim = qk_nope + qk_rope;
    const float *qh = q + (uint64_t)head * qk_dim;
    const float *low = qk_low + (uint64_t)head * kv_lora_dim;

    for (uint32_t j = threadIdx.x; j < kv_lora_dim; j += nth) low_sh[j] = low[j];
    for (uint32_t r = threadIdx.x; r < qk_rope; r += nth) q_rope_sh[r] = qh[qk_nope + r];
    __syncthreads();

    float corr0, corr1;
    glm_dev_rope_corr(qk_rope, n_ctx_orig, freq_base, beta_fast, beta_slow,
                      ext_factor, &corr0, &corr1);

    float local_max = -INFINITY;
    for (uint32_t s = threadIdx.x; s < n_selected; s += nth) {
        const uint32_t row = selected[s];
        float score = -INFINITY;
        if (row < cache_cap) {
            float dotv = 0.0f;
            const uint64_t lora_base = (uint64_t)row * kv_lora_dim;
            for (uint32_t j = 0; j < kv_lora_dim; j++) {
                dotv += low_sh[j] * glm_dev_cache_load<CF16>(kv_lora_cache, lora_base + j);
            }
            const uint64_t rope_base = (uint64_t)row * qk_rope;
            for (uint32_t r = 0; r < qk_rope; r += 2u) {
                float c, sn;
                glm_dev_rope_cs((float)row, r, qk_rope,
                                freq_base, freq_scale, ext_factor, attn_factor,
                                corr0, corr1, &c, &sn);
                const float x0 = glm_dev_cache_load<CF16>(k_rope_cache, rope_base + r);
                const float x1 = glm_dev_cache_load<CF16>(k_rope_cache, rope_base + r + 1u);
                dotv += q_rope_sh[r] * (x0 * c - x1 * sn) +
                        q_rope_sh[r + 1u] * (x0 * sn + x1 * c);
            }
            score = dotv * scale;
        }
        scores[s] = score;
        local_max = fmaxf(local_max, score);
    }
    red[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = nth >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
        __syncthreads();
    }
    const float max_score = red[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t s = threadIdx.x; s < n_selected; s += nth) {
        const float w = expf(scores[s] - max_score);
        scores[s] = w;
        local_sum += w;
    }
    red[threadIdx.x] = local_sum;
    __syncthreads();
    for (uint32_t stride = nth >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    const float denom = fmaxf(red[0], 1.0e-20f);
    __syncthreads();

    for (uint32_t j = threadIdx.x; j < kv_lora_dim; j += nth) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < n_selected; s++) {
            const uint32_t row = selected[s];
            if (row < cache_cap) {
                acc += scores[s] *
                       glm_dev_cache_load<CF16>(kv_lora_cache,
                                                (uint64_t)row * kv_lora_dim + j);
            }
        }
        lora_sum[j] = acc / denom;
    }
    __syncthreads();

    float *out = heads + (uint64_t)head * value_dim;
    for (uint32_t d = threadIdx.x; d < value_dim; d += nth) {
        const char *row = value_weight +
            ((uint64_t)head * value_dim + d) * value_row_bytes;
        out[d] = glm_dev_q8_row_dot_f32(row, lora_sum, kv_lora_dim);
    }
}

/* --- J/K: indexer scores. score(row) = sum_h max(dot(q[h], k_row)*scale,
 *        0) * weights[h]. One block per (row[, token]); the cache row is
 *        staged in shared memory and each warp handles a strided subset of
 *        heads with a shuffle reduction. --- */
template <bool CF16>
__global__ static void glm_indexer_score_one_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const char *key_cache,
        uint32_t n_rows,
        uint32_t n_head,
        uint32_t head_dim,
        float scale) {
    extern __shared__ float glm_is1_sh[];
    __shared__ float wsum[32];
    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t n_warps = blockDim.x >> 5;

    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        glm_is1_sh[d] = glm_dev_cache_load<CF16>(key_cache,
                                                 (uint64_t)row * head_dim + d);
    }
    __syncthreads();

    float acc = 0.0f;
    for (uint32_t h = warp; h < n_head; h += n_warps) {
        const float *qh = q + (uint64_t)h * head_dim;
        float dotv = 0.0f;
        for (uint32_t d = lane; d < head_dim; d += 32u) dotv += qh[d] * glm_is1_sh[d];
        dotv = warp_sum_f32(dotv);
        if (lane == 0) acc += fmaxf(dotv * scale, 0.0f) * weights[h];
    }
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (uint32_t w = 0; w < n_warps; w++) total += wsum[w];
        scores[row] = total;
    }
}

template <bool CF16>
__global__ static void glm_indexer_scores_batch_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const char *key_cache,
        uint32_t n_rows,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        float scale) {
    extern __shared__ float glm_isb_sh[];
    __shared__ float wsum[32];
    const uint32_t row = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (row >= n_rows || token >= n_tokens) return;
    float *dst = scores + (uint64_t)token * n_rows + row;
    const uint32_t visible = min(pos0 + token + 1u, n_rows);
    if (row >= visible) {
        if (threadIdx.x == 0) *dst = -INFINITY;
        return;
    }
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t n_warps = blockDim.x >> 5;

    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        glm_isb_sh[d] = glm_dev_cache_load<CF16>(key_cache,
                                                 (uint64_t)row * head_dim + d);
    }
    __syncthreads();

    const float *wt = weights + (uint64_t)token * n_head;
    float acc = 0.0f;
    for (uint32_t h = warp; h < n_head; h += n_warps) {
        const float *qh = q + ((uint64_t)token * n_head + h) * head_dim;
        float dotv = 0.0f;
        for (uint32_t d = lane; d < head_dim; d += 32u) dotv += qh[d] * glm_isb_sh[d];
        dotv = warp_sum_f32(dotv);
        if (lane == 0) acc += fmaxf(dotv * scale, 0.0f) * wt[h];
    }
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (uint32_t w = 0; w < n_warps; w++) total += wsum[w];
        *dst = total;
    }
}

extern "C" int ds4_gpu_add3_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const ds4_gpu_tensor *c,
        uint32_t                n) {
    if (!out || !a || !b || !c || n == 0 ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        a->bytes < (uint64_t)n * sizeof(float) ||
        b->bytes < (uint64_t)n * sizeof(float) ||
        c->bytes < (uint64_t)n * sizeof(float)) return 0;
    glm_add3_kernel<<<(n + 255) / 256, 256>>>(
            (float *)out->ptr,
            (const float *)a->ptr,
            (const float *)b->ptr,
            (const float *)c->ptr,
            n);
    return cuda_ok(cudaGetLastError(), "glm add3 launch");
}

extern "C" int ds4_gpu_add_rms_norm_weight_tensor(
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *sum_out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps) {
    if (!norm_out || !sum_out || !a || !b || !model_map || n == 0 ||
        norm_out->bytes < (uint64_t)n * sizeof(float) ||
        sum_out->bytes < (uint64_t)n * sizeof(float) ||
        a->bytes < (uint64_t)n * sizeof(float) ||
        b->bytes < (uint64_t)n * sizeof(float)) return 0;
    const uint64_t w_bytes = (uint64_t)n * sizeof(float);
    if (weight_offset > model_size || w_bytes > model_size - weight_offset) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, w_bytes, "glm_add_rms_weight");
    if (!wptr) return 0;
    glm_add_rms_norm_weight_kernel<<<1, 256>>>(
            (float *)norm_out->ptr,
            (float *)sum_out->ptr,
            (const float *)a->ptr,
            (const float *)b->ptr,
            (const float *)wptr,
            n,
            eps);
    return cuda_ok(cudaGetLastError(), "glm add_rms_norm_weight launch");
}

extern "C" int ds4_gpu_embed_token_q8_0_tensor(
        ds4_gpu_tensor *out,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd) {
    if (!out || !model_map || n_vocab == 0 || n_embd == 0 || token >= n_vocab ||
        out->bytes < (uint64_t)n_embd * sizeof(float)) return 0;
    const uint64_t row_blocks = ((uint64_t)n_embd + 31u) / 32u;
    const uint64_t row_bytes = row_blocks * 34u;
    const uint64_t table_bytes = (uint64_t)n_vocab * row_bytes;
    if (weight_offset > model_size || table_bytes > model_size - weight_offset) return 0;
    const char *w = cuda_model_range_ptr(model_map, weight_offset, table_bytes, "glm_embed_q8_0");
    if (!w) return 0;
    const unsigned char *row = (const unsigned char *)w + (uint64_t)token * row_bytes;
    dequant_q8_0_to_f32_kernel<<<(n_embd + 255) / 256, 256>>>(
            (float *)out->ptr,
            row,
            (uint64_t)n_embd,
            1,
            row_blocks);
    return cuda_ok(cudaGetLastError(), "glm embed_token_q8_0 launch");
}

extern "C" int ds4_gpu_embed_tokens_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd) {
    if (!out || !tokens || !model_map || n_vocab == 0 || n_tokens == 0 || n_embd == 0 ||
        out->bytes < (uint64_t)n_tokens * n_embd * sizeof(float) ||
        tokens->bytes < (uint64_t)n_tokens * sizeof(int32_t)) return 0;
    const uint64_t row_blocks = ((uint64_t)n_embd + 31u) / 32u;
    const uint64_t row_bytes = row_blocks * 34u;
    const uint64_t table_bytes = (uint64_t)n_vocab * row_bytes;
    if (weight_offset > model_size || table_bytes > model_size - weight_offset) return 0;
    const char *w = cuda_model_range_ptr(model_map, weight_offset, table_bytes, "glm_embed_q8_0");
    if (!w) return 0;
    const uint64_t n = (uint64_t)n_tokens * n_embd;
    glm_embed_tokens_q8_0_kernel<<<(unsigned)((n + 255) / 256), 256>>>(
            (float *)out->ptr,
            (const int32_t *)tokens->ptr,
            (const unsigned char *)w,
            n_vocab,
            n_tokens,
            n_embd,
            row_blocks);
    return cuda_ok(cudaGetLastError(), "glm embed_tokens_q8_0 launch");
}

extern "C" int ds4_gpu_flush_encoder(void) {
    /* Metal closes the open compute encoder here so later encoders see the
     * writes of earlier ones inside a batch (it does NOT commit). CUDA
     * default-stream launches are already ordered, so this is a no-op. */
    return 1;
}

extern "C" int ds4_gpu_glm_attention_flash_staged_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16) {
    ds4_cuda_glm_stub_note("ds4_gpu_glm_attention_flash_staged_tensor");
    return 0;
}

extern "C" int ds4_gpu_glm_attention_flash_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16) {
    ds4_cuda_glm_stub_note("ds4_gpu_glm_attention_flash_tensor");
    return 0;
}

extern "C" int ds4_gpu_glm_attention_full_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16) {
    const uint64_t esize = cache_f16 ? sizeof(__half) : sizeof(float);
    if (!heads || !q || !key_cache || !value_cache ||
        n_tokens == 0 || cache_len == 0 || n_head == 0 ||
        qk_dim == 0 || value_dim == 0 ||
        qk_dim > GLM_CUDA_ATTN_QK_MAX ||
        value_dim > 256u * GLM_CUDA_ATTN_VACC_MAX ||
        cache_len > cache_cap || n_head > 65535u ||
        q->bytes < (uint64_t)n_tokens * n_head * qk_dim * sizeof(float) ||
        heads->bytes < (uint64_t)n_tokens * n_head * value_dim * sizeof(float) ||
        key_cache->bytes < (uint64_t)cache_len * n_head * qk_dim * esize ||
        value_cache->bytes < (uint64_t)cache_len * n_head * value_dim * esize) {
        return 0;
    }
    /* Scale computed like the CPU reference / Metal host: 1/sqrt(qk_dim). */
    const float scale = 1.0f / sqrtf((float)qk_dim);
    dim3 grid(n_tokens, n_head, 1);
    if (cache_f16) {
        glm_attention_full_kernel<true><<<grid, 256>>>(
                (float *)heads->ptr,
                (const float *)q->ptr,
                (const char *)key_cache->ptr,
                (const char *)value_cache->ptr,
                pos0, n_tokens, cache_len,
                n_head, qk_dim, value_dim, scale);
    } else {
        glm_attention_full_kernel<false><<<grid, 256>>>(
                (float *)heads->ptr,
                (const float *)q->ptr,
                (const char *)key_cache->ptr,
                (const char *)value_cache->ptr,
                pos0, n_tokens, cache_len,
                n_head, qk_dim, value_dim, scale);
    }
    return cuda_ok(cudaGetLastError(), "glm attention_full launch");
}

extern "C" int ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    ds4_cuda_glm_stub_note("ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor");
    return 0;
}

extern "C" int ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    ds4_cuda_glm_stub_note("ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor");
    return 0;
}

extern "C" int ds4_gpu_glm_attention_indexed_batch_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    ds4_cuda_glm_stub_note("ds4_gpu_glm_attention_indexed_batch_tensor");
    return 0;
}

extern "C" int ds4_gpu_glm_attention_indexed_decode_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    const uint64_t esize = cache_f16 ? sizeof(__half) : sizeof(float);
    const uint32_t qk_dim = qk_nope + qk_rope;
    if (!heads || !q || !qk_low || !kv_lora_cache || !k_rope_cache ||
        !model_map || !selected ||
        n_selected == 0 || cache_cap == 0 || n_head == 0 ||
        kv_lora_dim == 0 || value_dim == 0 ||
        qk_rope == 0 || (qk_rope & 1u) != 0 ||
        heads->bytes < (uint64_t)n_head * value_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_head * qk_dim * sizeof(float) ||
        qk_low->bytes < (uint64_t)n_head * kv_lora_dim * sizeof(float) ||
        selected->bytes < (uint64_t)n_selected * sizeof(uint32_t) ||
        kv_lora_cache->bytes < (uint64_t)cache_cap * kv_lora_dim * esize ||
        k_rope_cache->bytes < (uint64_t)cache_cap * qk_rope * esize) {
        return 0;
    }
    const uint64_t value_row_bytes = glm_cuda_q8_row_bytes(kv_lora_dim);
    const uint64_t weight_bytes = (uint64_t)n_head * value_dim * value_row_bytes;
    if (value_weight_offset > model_size ||
        weight_bytes > model_size - value_weight_offset) {
        return 0;
    }
    const char *v_w = cuda_model_range_ptr(model_map, value_weight_offset,
                                           weight_bytes, "glm_attn_v_b");
    if (!v_w) return 0;
    const uint64_t shmem =
        ((uint64_t)n_selected + 2u * kv_lora_dim + qk_rope + 256u) * sizeof(float);
    /* scores(2048) + lora_sum(512) + low(512) + q_rope(64) + red(256) =
     * 13.25 KiB at GLM-5.2 defaults — well under the 48 KiB static limit.
     * Larger n_selected opts into >48 KiB dynamic shared when the device
     * allows it (V100: 96 KiB). */
    if (cache_f16) {
        if (!glm_cuda_ensure_dyn_shmem(
                    (const void *)glm_attention_indexed_decode_kernel<true>, shmem)) {
            return 0;
        }
    } else {
        if (!glm_cuda_ensure_dyn_shmem(
                    (const void *)glm_attention_indexed_decode_kernel<false>, shmem)) {
            return 0;
        }
    }
    /* Scale computed like the CPU reference / Metal host: 1/sqrt(qk_dim). */
    const float scale = 1.0f / sqrtf((float)qk_dim);
    if (cache_f16) {
        glm_attention_indexed_decode_kernel<true><<<n_head, 256, (size_t)shmem>>>(
                (float *)heads->ptr,
                (const float *)q->ptr,
                (const float *)qk_low->ptr,
                (const char *)kv_lora_cache->ptr,
                (const char *)k_rope_cache->ptr,
                v_w,
                (const uint32_t *)selected->ptr,
                n_selected, cache_cap, n_head,
                kv_lora_dim, qk_nope, qk_rope, value_dim,
                value_row_bytes, n_ctx_orig, scale,
                freq_base, freq_scale, ext_factor, attn_factor,
                beta_fast, beta_slow);
    } else {
        glm_attention_indexed_decode_kernel<false><<<n_head, 256, (size_t)shmem>>>(
                (float *)heads->ptr,
                (const float *)q->ptr,
                (const float *)qk_low->ptr,
                (const char *)kv_lora_cache->ptr,
                (const char *)k_rope_cache->ptr,
                v_w,
                (const uint32_t *)selected->ptr,
                n_selected, cache_cap, n_head,
                kv_lora_dim, qk_nope, qk_rope, value_dim,
                value_row_bytes, n_ctx_orig, scale,
                freq_base, freq_scale, ext_factor, attn_factor,
                beta_fast, beta_slow);
    }
    return cuda_ok(cudaGetLastError(), "glm attention_indexed_decode launch");
}

extern "C" int ds4_gpu_glm_build_kv_cache_flash_tensor(
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *kv_raw,
        const ds4_gpu_tensor *k_nope,
        const ds4_gpu_tensor *value,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16) {
    ds4_cuda_glm_stub_note("ds4_gpu_glm_build_kv_cache_flash_tensor");
    return 0;
}

extern "C" int ds4_gpu_glm_build_kv_cache_tensor(
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *kv_raw,
        const ds4_gpu_tensor *k_nope,
        const ds4_gpu_tensor *value,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16) {
    const uint64_t esize = cache_f16 ? sizeof(__half) : sizeof(float);
    const uint32_t qk_dim = qk_nope + qk_rope;
    if (!key_cache || !value_cache || !kv_raw || !k_nope || !value ||
        n_tokens == 0 || cache_cap == 0 || n_head == 0 || n_head > 65535u ||
        qk_nope == 0 || qk_rope == 0 || (qk_rope & 1u) != 0 ||
        value_dim == 0 ||
        kv_lora_dim + qk_rope > kv_raw_dim ||
        kv_raw->bytes < (uint64_t)n_tokens * kv_raw_dim * sizeof(float) ||
        k_nope->bytes < (uint64_t)n_tokens * n_head * qk_nope * sizeof(float) ||
        value->bytes < (uint64_t)n_tokens * n_head * value_dim * sizeof(float) ||
        key_cache->bytes < (uint64_t)cache_cap * n_head * qk_dim * esize ||
        value_cache->bytes < (uint64_t)cache_cap * n_head * value_dim * esize) {
        return 0;
    }
    dim3 grid(n_tokens, n_head, 1);
    if (cache_f16) {
        glm_build_kv_cache_kernel<true><<<grid, 256>>>(
                (char *)key_cache->ptr,
                (char *)value_cache->ptr,
                (const float *)kv_raw->ptr,
                (const float *)k_nope->ptr,
                (const float *)value->ptr,
                pos0, n_tokens, cache_cap, n_head,
                kv_raw_dim, kv_lora_dim, qk_nope, qk_rope, value_dim,
                n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                beta_fast, beta_slow);
    } else {
        glm_build_kv_cache_kernel<false><<<grid, 256>>>(
                (char *)key_cache->ptr,
                (char *)value_cache->ptr,
                (const float *)kv_raw->ptr,
                (const float *)k_nope->ptr,
                (const float *)value->ptr,
                pos0, n_tokens, cache_cap, n_head,
                kv_raw_dim, kv_lora_dim, qk_nope, qk_rope, value_dim,
                n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
                beta_fast, beta_slow);
    }
    return cuda_ok(cudaGetLastError(), "glm build_kv_cache launch");
}

extern "C" int ds4_gpu_glm_fill_selected_range_batch_tensor(
        ds4_gpu_tensor *selected,
        uint32_t        n_tokens,
        uint32_t        pos0,
        uint32_t        n_selected,
        uint32_t        pad_row) {
    if (!selected || n_tokens == 0 || n_selected == 0 ||
        selected->bytes < (uint64_t)n_tokens * n_selected * sizeof(uint32_t)) return 0;
    const uint32_t total = n_tokens * n_selected;
    glm_fill_selected_range_batch_kernel<<<(total + 255) / 256, 256>>>(
            (uint32_t *)selected->ptr,
            n_tokens,
            pos0,
            n_selected,
            pad_row);
    return cuda_ok(cudaGetLastError(), "glm fill_selected_range_batch launch");
}

extern "C" int ds4_gpu_glm_fill_selected_range_tensor(
        ds4_gpu_tensor *selected,
        uint32_t        n_selected) {
    if (!selected || n_selected == 0 ||
        selected->bytes < (uint64_t)n_selected * sizeof(uint32_t)) return 0;
    glm_fill_selected_range_kernel<<<(n_selected + 255) / 256, 256>>>(
            (uint32_t *)selected->ptr,
            n_selected);
    return cuda_ok(cudaGetLastError(), "glm fill_selected_range launch");
}

extern "C" int ds4_gpu_glm_indexer_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        rot_dim,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow) {
    /* Rotates the FIRST rot_dim of each head (rot_offset = 0). */
    return glm_rope_offset_launch(x, n_tokens, n_head, head_dim, rot_dim,
                                  0u, pos0, n_ctx_orig,
                                  freq_base, freq_scale, ext_factor,
                                  attn_factor, beta_fast, beta_slow,
                                  "glm indexer_rope_tail launch");
}

extern "C" int ds4_gpu_glm_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t              n_rows,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale,
        bool                  cache_f16) {
    const uint64_t esize = cache_f16 ? sizeof(__half) : sizeof(float);
    if (!scores || !q || !weights || !indexer_key_cache ||
        n_rows == 0 || n_head == 0 || head_dim == 0 ||
        scores->bytes < (uint64_t)n_rows * sizeof(float) ||
        q->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        weights->bytes < (uint64_t)n_head * sizeof(float) ||
        indexer_key_cache->bytes < (uint64_t)n_rows * head_dim * esize) {
        return 0;
    }
    const uint64_t shmem = (uint64_t)head_dim * sizeof(float);
    if (shmem > 48u * 1024u) return 0;
    if (cache_f16) {
        glm_indexer_score_one_kernel<true><<<n_rows, 256, (size_t)shmem>>>(
                (float *)scores->ptr,
                (const float *)q->ptr,
                (const float *)weights->ptr,
                (const char *)indexer_key_cache->ptr,
                n_rows, n_head, head_dim, scale);
    } else {
        glm_indexer_score_one_kernel<false><<<n_rows, 256, (size_t)shmem>>>(
                (float *)scores->ptr,
                (const float *)q->ptr,
                (const float *)weights->ptr,
                (const char *)indexer_key_cache->ptr,
                n_rows, n_head, head_dim, scale);
    }
    return cuda_ok(cudaGetLastError(), "glm indexer_score_one launch");
}

extern "C" int ds4_gpu_glm_indexer_scores_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t              n_rows,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale,
        bool                  cache_f16) {
    const uint64_t esize = cache_f16 ? sizeof(__half) : sizeof(float);
    if (!scores || !q || !weights || !indexer_key_cache ||
        n_rows == 0 || n_tokens == 0 || n_tokens > 65535u ||
        n_head == 0 || head_dim == 0 ||
        scores->bytes < (uint64_t)n_tokens * n_rows * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        weights->bytes < (uint64_t)n_tokens * n_head * sizeof(float) ||
        indexer_key_cache->bytes < (uint64_t)n_rows * head_dim * esize) {
        return 0;
    }
    const uint64_t shmem = (uint64_t)head_dim * sizeof(float);
    if (shmem > 48u * 1024u) return 0;
    dim3 grid(n_rows, n_tokens, 1);
    if (cache_f16) {
        glm_indexer_scores_batch_kernel<true><<<grid, 256, (size_t)shmem>>>(
                (float *)scores->ptr,
                (const float *)q->ptr,
                (const float *)weights->ptr,
                (const char *)indexer_key_cache->ptr,
                n_rows, n_tokens, pos0, n_head, head_dim, scale);
    } else {
        glm_indexer_scores_batch_kernel<false><<<grid, 256, (size_t)shmem>>>(
                (float *)scores->ptr,
                (const float *)q->ptr,
                (const float *)weights->ptr,
                (const char *)indexer_key_cache->ptr,
                n_rows, n_tokens, pos0, n_head, head_dim, scale);
    }
    return cuda_ok(cudaGetLastError(), "glm indexer_scores_batch launch");
}

extern "C" int ds4_gpu_glm_k_b_project_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *kv_norm,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              n_head) {
    if (!out || !kv_norm || !model_map ||
        n_tokens == 0 || kv_lora_dim == 0 || qk_nope == 0 || n_head == 0 ||
        n_head > 65535u ||
        out->bytes < (uint64_t)n_tokens * n_head * qk_nope * sizeof(float) ||
        kv_norm->bytes < (uint64_t)n_tokens * kv_lora_dim * sizeof(float)) {
        return 0;
    }
    const uint32_t threads = ((qk_nope + 31u) / 32u) * 32u;
    const uint64_t shmem = (uint64_t)kv_lora_dim * sizeof(float);
    if (threads > 1024u || shmem > 48u * 1024u) return 0;
    const uint64_t row_bytes = glm_cuda_q8_row_bytes(qk_nope);
    const uint64_t weight_bytes = (uint64_t)n_head * kv_lora_dim * row_bytes;
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    const char *w = cuda_model_range_ptr(model_map, weight_offset,
                                         weight_bytes, "glm_attn_k_b");
    if (!w) return 0;
    dim3 grid(n_tokens, n_head, 1);
    glm_k_b_project_q8_0_kernel<<<grid, threads, (size_t)shmem>>>(
            (float *)out->ptr,
            w,
            (const float *)kv_norm->ptr,
            n_tokens, kv_lora_dim, qk_nope, n_head, row_bytes);
    return cuda_ok(cudaGetLastError(), "glm k_b_project launch");
}

extern "C" int ds4_gpu_glm_kv_lora_rms_norm_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *kv_raw,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        float                 eps) {
    if (!out || !kv_raw || !model_map || n_tokens == 0 ||
        kv_lora_dim == 0 || kv_lora_dim > kv_raw_dim ||
        out->bytes < (uint64_t)n_tokens * kv_lora_dim * sizeof(float) ||
        kv_raw->bytes < (uint64_t)n_tokens * kv_raw_dim * sizeof(float)) return 0;
    const uint64_t w_bytes = (uint64_t)kv_lora_dim * sizeof(float);
    if (weight_offset > model_size || w_bytes > model_size - weight_offset) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, w_bytes, "glm_kv_lora_rms_weight");
    if (!wptr) return 0;
    glm_kv_lora_rms_norm_kernel<<<n_tokens, 256>>>(
            (float *)out->ptr,
            (const float *)kv_raw->ptr,
            (const float *)wptr,
            n_tokens,
            kv_raw_dim,
            kv_lora_dim,
            eps);
    return cuda_ok(cudaGetLastError(), "glm kv_lora_rms_norm launch");
}

extern "C" int ds4_gpu_glm_qk_lowrank_q8_0_batch_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim) {
    ds4_cuda_glm_stub_note("ds4_gpu_glm_qk_lowrank_q8_0_batch_tensor");
    return 0;
}

extern "C" int ds4_gpu_glm_qk_lowrank_q8_0_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim) {
    if (!qk_low || !q || !model_map ||
        n_head == 0 || kv_lora_dim == 0 || qk_nope == 0 ||
        qk_nope > qk_dim ||
        qk_low->bytes < (uint64_t)n_head * kv_lora_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_head * qk_dim * sizeof(float)) {
        return 0;
    }
    const uint64_t shmem = (uint64_t)qk_nope * sizeof(float);
    if (shmem > 48u * 1024u) return 0;
    const uint64_t row_bytes = glm_cuda_q8_row_bytes(qk_nope);
    const uint64_t weight_bytes = (uint64_t)n_head * kv_lora_dim * row_bytes;
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    const char *w = cuda_model_range_ptr(model_map, weight_offset,
                                         weight_bytes, "glm_attn_k_b");
    if (!w) return 0;
    glm_qk_lowrank_q8_0_kernel<<<n_head, 256, (size_t)shmem>>>(
            (float *)qk_low->ptr,
            w,
            (const float *)q->ptr,
            n_head, kv_lora_dim, qk_nope, qk_dim, row_bytes);
    return cuda_ok(cudaGetLastError(), "glm qk_lowrank launch");
}

extern "C" int ds4_gpu_glm_qkv_norm_store_compact_kv_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_weight_offset,
        uint32_t              q_n,
        ds4_gpu_tensor       *kv_lora_cache,
        ds4_gpu_tensor       *k_rope_cache,
        const ds4_gpu_tensor *kv_raw,
        uint64_t              kv_weight_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_rope,
        bool                  cache_f16,
        float                 eps) {
    const uint64_t esize = cache_f16 ? sizeof(__half) : sizeof(float);
    if (!q_out || !q || !model_map || !kv_lora_cache || !k_rope_cache || !kv_raw ||
        n_tokens == 0 || cache_cap == 0 || q_n == 0 ||
        kv_lora_dim == 0 || qk_rope == 0 ||
        kv_lora_dim + qk_rope > kv_raw_dim ||
        q_out->bytes < (uint64_t)n_tokens * q_n * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * q_n * sizeof(float) ||
        kv_raw->bytes < (uint64_t)n_tokens * kv_raw_dim * sizeof(float) ||
        kv_lora_cache->bytes < (uint64_t)cache_cap * kv_lora_dim * esize ||
        k_rope_cache->bytes < (uint64_t)cache_cap * qk_rope * esize) {
        return 0;
    }
    const uint64_t q_w_bytes = (uint64_t)q_n * sizeof(float);
    const uint64_t kv_w_bytes = (uint64_t)kv_lora_dim * sizeof(float);
    if (q_weight_offset > model_size || q_w_bytes > model_size - q_weight_offset) return 0;
    if (kv_weight_offset > model_size || kv_w_bytes > model_size - kv_weight_offset) return 0;
    const float *q_w = (const float *)cuda_model_range_ptr(model_map, q_weight_offset,
                                                           q_w_bytes, "glm_q_a_norm");
    const float *kv_w = (const float *)cuda_model_range_ptr(model_map, kv_weight_offset,
                                                            kv_w_bytes, "glm_kv_a_norm");
    if (!q_w || !kv_w) return 0;

    /* p0: plain RMS norm of q (row width q_n) — reuse the kv_lora norm
     * kernel with raw_dim == lora_dim == q_n. */
    glm_kv_lora_rms_norm_kernel<<<n_tokens, 256>>>(
            (float *)q_out->ptr,
            (const float *)q->ptr,
            q_w,
            n_tokens, q_n, q_n, eps);
    if (!cuda_ok(cudaGetLastError(), "glm qkv_norm_store q-norm launch")) return 0;

    /* p1: RMS norm kv_raw[0..kv_lora_dim) into kv_lora_cache[pos]. */
    if (cache_f16) {
        glm_kv_norm_store_cache_kernel<true><<<n_tokens, 256>>>(
                (char *)kv_lora_cache->ptr,
                (const float *)kv_raw->ptr,
                kv_w,
                pos0, n_tokens, cache_cap, kv_raw_dim, kv_lora_dim, eps);
    } else {
        glm_kv_norm_store_cache_kernel<false><<<n_tokens, 256>>>(
                (char *)kv_lora_cache->ptr,
                (const float *)kv_raw->ptr,
                kv_w,
                pos0, n_tokens, cache_cap, kv_raw_dim, kv_lora_dim, eps);
    }
    if (!cuda_ok(cudaGetLastError(), "glm qkv_norm_store kv-norm launch")) return 0;

    /* p2: raw (unrotated) rope tail into k_rope_cache[pos]. */
    const uint64_t total = (uint64_t)n_tokens * qk_rope;
    if (cache_f16) {
        glm_store_rope_tail_kernel<true><<<(unsigned)((total + 255u) / 256u), 256>>>(
                (char *)k_rope_cache->ptr,
                (const float *)kv_raw->ptr,
                pos0, n_tokens, cache_cap, kv_raw_dim, kv_lora_dim, qk_rope);
    } else {
        glm_store_rope_tail_kernel<false><<<(unsigned)((total + 255u) / 256u), 256>>>(
                (char *)k_rope_cache->ptr,
                (const float *)kv_raw->ptr,
                pos0, n_tokens, cache_cap, kv_raw_dim, kv_lora_dim, qk_rope);
    }
    return cuda_ok(cudaGetLastError(), "glm qkv_norm_store rope-tail launch");
}

extern "C" int ds4_gpu_glm_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        rot_dim,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow) {
    /* Tail rotation: the rotated slice is the LAST rot_dim of each head. */
    if (rot_dim > head_dim) return 0;
    return glm_rope_offset_launch(x, n_tokens, n_head, head_dim, rot_dim,
                                  head_dim - rot_dim, pos0, n_ctx_orig,
                                  freq_base, freq_scale, ext_factor,
                                  attn_factor, beta_fast, beta_slow,
                                  "glm rope_tail launch");
}

extern "C" int ds4_gpu_glm_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride) {
    return glm_routed_moe_launch(out, mid, model_map, model_size,
                                 gate_offset, up_offset, down_offset,
                                 gate_type, up_type, down_type,
                                 gate_expert_bytes, gate_row_bytes,
                                 up_expert_bytes, up_row_bytes,
                                 down_expert_bytes, down_row_bytes,
                                 expert_in_dim, expert_mid_dim, out_dim,
                                 selected, weights, n_total_expert, n_expert,
                                 layer_index, x, n_tokens,
                                 (uint64_t)mid_token_stride);
}

extern "C" int ds4_gpu_glm_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        bool                    force_resident) {
    /* MVP: the launcher always addresses the experts through the model
     * mapping, so force_resident and streaming mode share one path. */
    (void)force_resident;
    return glm_routed_moe_launch(out, mid, model_map, model_size,
                                 gate_offset, up_offset, down_offset,
                                 gate_type, up_type, down_type,
                                 gate_expert_bytes, gate_row_bytes,
                                 up_expert_bytes, up_row_bytes,
                                 down_expert_bytes, down_row_bytes,
                                 expert_in_dim, expert_mid_dim, out_dim,
                                 selected, weights, n_total_expert, n_expert,
                                 layer_index, x, 1u,
                                 (uint64_t)n_expert * expert_mid_dim);
}

extern "C" int ds4_gpu_glm_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens) {
    return glm_router_select_launch(selected, weights, probs,
                                    model_map, model_size, bias_offset,
                                    logits, n_expert, n_expert_used,
                                    expert_weight_scale, n_tokens);
}

extern "C" int ds4_gpu_glm_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale) {
    return glm_router_select_launch(selected, weights, probs,
                                    model_map, model_size, bias_offset,
                                    logits, n_expert, n_expert_used,
                                    expert_weight_scale, 1u);
}

extern "C" int ds4_gpu_glm_store_compact_kv_tensor(
        ds4_gpu_tensor       *kv_lora_cache,
        ds4_gpu_tensor       *k_rope_cache,
        const ds4_gpu_tensor *kv_norm,
        const ds4_gpu_tensor *kv_raw,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_rope,
        bool                  cache_f16) {
    const uint64_t esize = cache_f16 ? sizeof(__half) : sizeof(float);
    if (!kv_lora_cache || !k_rope_cache || !kv_norm || !kv_raw ||
        n_tokens == 0 || cache_cap == 0 ||
        kv_lora_dim == 0 || qk_rope == 0 ||
        kv_lora_dim + qk_rope > kv_raw_dim ||
        kv_norm->bytes < (uint64_t)n_tokens * kv_lora_dim * sizeof(float) ||
        kv_raw->bytes < (uint64_t)n_tokens * kv_raw_dim * sizeof(float) ||
        kv_lora_cache->bytes < (uint64_t)cache_cap * kv_lora_dim * esize ||
        k_rope_cache->bytes < (uint64_t)cache_cap * qk_rope * esize) {
        return 0;
    }
    const uint64_t total = (uint64_t)n_tokens * (kv_lora_dim + qk_rope);
    if (cache_f16) {
        glm_store_compact_kv_kernel<true><<<(unsigned)((total + 255u) / 256u), 256>>>(
                (char *)kv_lora_cache->ptr,
                (char *)k_rope_cache->ptr,
                (const float *)kv_norm->ptr,
                (const float *)kv_raw->ptr,
                pos0, n_tokens, cache_cap, kv_raw_dim, kv_lora_dim, qk_rope);
    } else {
        glm_store_compact_kv_kernel<false><<<(unsigned)((total + 255u) / 256u), 256>>>(
                (char *)kv_lora_cache->ptr,
                (char *)k_rope_cache->ptr,
                (const float *)kv_norm->ptr,
                (const float *)kv_raw->ptr,
                pos0, n_tokens, cache_cap, kv_raw_dim, kv_lora_dim, qk_rope);
    }
    return cuda_ok(cudaGetLastError(), "glm store_compact_kv launch");
}

extern "C" int ds4_gpu_glm_store_indexer_k_tensor(
        ds4_gpu_tensor       *indexer_key_cache,
        const ds4_gpu_tensor *raw_k,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              bias_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              head_dim,
        uint32_t              rot_dim,
        uint32_t              n_ctx_orig,
        float                 eps,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16) {
    const uint64_t esize = cache_f16 ? sizeof(__half) : sizeof(float);
    if (!indexer_key_cache || !raw_k || !model_map ||
        n_tokens == 0 || cache_cap == 0 || head_dim == 0 ||
        rot_dim == 0 || (rot_dim & 1u) != 0 || rot_dim > head_dim ||
        raw_k->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        indexer_key_cache->bytes < (uint64_t)cache_cap * head_dim * esize) {
        return 0;
    }
    const uint64_t wb = (uint64_t)head_dim * sizeof(float);
    if (weight_offset > model_size || wb > model_size - weight_offset) return 0;
    if (bias_offset > model_size || wb > model_size - bias_offset) return 0;
    const float *w = (const float *)cuda_model_range_ptr(model_map, weight_offset,
                                                         wb, "glm_indexer_k_norm");
    const float *b = (const float *)cuda_model_range_ptr(model_map, bias_offset,
                                                         wb, "glm_indexer_k_norm_b");
    if (!w || !b) return 0;
    if (cache_f16) {
        glm_store_indexer_k_kernel<true><<<n_tokens, 128>>>(
                (char *)indexer_key_cache->ptr,
                (const float *)raw_k->ptr,
                w, b,
                pos0, n_tokens, cache_cap, head_dim, rot_dim, n_ctx_orig,
                eps, freq_base, freq_scale, ext_factor, attn_factor,
                beta_fast, beta_slow);
    } else {
        glm_store_indexer_k_kernel<false><<<n_tokens, 128>>>(
                (char *)indexer_key_cache->ptr,
                (const float *)raw_k->ptr,
                w, b,
                pos0, n_tokens, cache_cap, head_dim, rot_dim, n_ctx_orig,
                eps, freq_base, freq_scale, ext_factor, attn_factor,
                beta_fast, beta_slow);
    }
    return cuda_ok(cudaGetLastError(), "glm store_indexer_k launch");
}

extern "C" int ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor              *selected,
        uint32_t                           n_selected) {
    if (!g_ssd_streaming_mode) return 1;
    if (!table || !selected || n_selected == 0 ||
        selected->bytes < (uint64_t)n_selected * sizeof(int32_t)) {
        return 0;
    }
    int32_t stack_ids[GLM_CUDA_MAX_TOPK];
    int32_t *ids = stack_ids;
    int32_t *heap_ids = NULL;
    if (n_selected > GLM_CUDA_MAX_TOPK) {
        heap_ids = (int32_t *)malloc((size_t)n_selected * sizeof(int32_t));
        if (!heap_ids) return 0;
        ids = heap_ids;
    }
    int ok = ds4_gpu_tensor_read(selected, 0, ids, (uint64_t)n_selected * sizeof(int32_t));
    if (ok) ok = ds4_gpu_stream_expert_cache_begin_selected_load(table, ids, n_selected);
    free(heap_ids);
    return ok;
}

extern "C" int ds4_gpu_glm_value_project_q8_0_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *lora,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              value_dim) {
    if (!heads || !lora || !model_map ||
        n_tokens == 0 || n_tokens > 65535u || n_head == 0 ||
        kv_lora_dim == 0 || value_dim == 0 ||
        heads->bytes < (uint64_t)n_tokens * n_head * value_dim * sizeof(float) ||
        lora->bytes < (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float)) {
        return 0;
    }
    const uint64_t shmem = (uint64_t)kv_lora_dim * sizeof(float);
    if (shmem > 48u * 1024u) return 0;
    const uint64_t row_bytes = glm_cuda_q8_row_bytes(kv_lora_dim);
    const uint64_t weight_bytes = (uint64_t)n_head * value_dim * row_bytes;
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    const char *w = cuda_model_range_ptr(model_map, weight_offset,
                                         weight_bytes, "glm_attn_v_b");
    if (!w) return 0;
    dim3 grid(n_head, n_tokens, 1);
    glm_value_project_q8_0_batch_heads_kernel<<<grid, 256, (size_t)shmem>>>(
            (float *)heads->ptr,
            w,
            (const float *)lora->ptr,
            n_tokens, n_head, kv_lora_dim, value_dim, row_bytes);
    return cuda_ok(cudaGetLastError(), "glm value_project_batch_heads launch");
}

extern "C" int ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    /* "decode mpp" + "model view" are Metal-only performance selectors;
     * numerics are identical to the plain q8_0 matmul. */
    return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size, weight_offset,
                                      in_dim, out_dim, x, n_tok);
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    /* "model view" is a Metal-only weight-addressing selector; numerics are
     * identical to the existing fused single-token gate+up+swiglu path. */
    return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(gate, up, mid,
                                                     model_map, model_size,
                                                     gate_offset, up_offset,
                                                     in_dim, out_dim,
                                                     x, clamp);
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp) {
    /* Multi-token variant: gate/up/mid are [n_tok x out_dim] with rows packed
     * contiguously, so one flat swiglu over out_dim*n_tok elements matches the
     * per-row CPU reference (swiglu, ds4.c). */
    if (!gate || !up || !mid || !x || !model_map ||
        in_dim == 0 || out_dim == 0 || n_tok == 0 ||
        out_dim * n_tok > UINT32_MAX) return 0;
    return ds4_gpu_matmul_q8_0_pair_tensor(gate, up,
                                           model_map, model_size,
                                           gate_offset, up_offset,
                                           in_dim, out_dim, out_dim,
                                           x, n_tok) &&
           ds4_gpu_swiglu_tensor(mid, gate, up,
                                 (uint32_t)(out_dim * n_tok), clamp, 1.0f);
}

extern "C" int ds4_gpu_shared_mid_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    /* Mid-only variant of the fused shared-expert gate+up+swiglu: gate/up go
     * to a private device scratch (separate from cuda_tmp_alloc's arena,
     * which the pair matmul uses for prequant), only mid is an output. */
    if (!mid || !x || !model_map || in_dim == 0 || out_dim == 0 ||
        out_dim > UINT32_MAX ||
        mid->bytes < out_dim * sizeof(float) ||
        x->bytes < in_dim * sizeof(float)) return 0;
    const uint64_t half_bytes = out_dim * sizeof(float);
    void *scratch = glm_swiglu_scratch_alloc(2u * half_bytes, "glm shared mid swiglu");
    if (!scratch) return 0;
    ds4_gpu_tensor gate_t;
    gate_t.ptr = scratch;
    gate_t.bytes = half_bytes;
    gate_t.owner = 0;
    ds4_gpu_tensor up_t;
    up_t.ptr = (char *)scratch + half_bytes;
    up_t.bytes = half_bytes;
    up_t.owner = 0;
    return ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(&gate_t, &up_t, mid,
                                                     model_map, model_size,
                                                     gate_offset, up_offset,
                                                     in_dim, out_dim,
                                                     x, clamp);
}
