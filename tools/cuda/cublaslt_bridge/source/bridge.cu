// Optional C bridge over our measured transforms and NVIDIA's documented cuBLASLt API.
#define TF_LT_BUILD
#include "tf_cublaslt_bridge.h"
#include "validation.hpp"
#include "tune_storage.hpp"
#include "kernels.cuh"
#include "fp8_extra.cuh"
#include <cuda.h>
#include <cublasLt.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>

namespace {
thread_local char creation_error[512] = "";
struct Failure { tf_lt_status status; std::string text; };
void runtime(cudaError_t error) {
    if (error != cudaSuccess) throw Failure{TF_LT_CUDA_ERROR, cudaGetErrorString(error)};
}
void driver(CUresult error) {
    if (error != CUDA_SUCCESS) {
        const char* text = nullptr; cuGetErrorString(error, &text);
        throw Failure{TF_LT_CUDA_ERROR, text ? text : "CUDA driver error"};
    }
}
void lt(cublasStatus_t error) {
    if (error != CUBLAS_STATUS_SUCCESS) throw Failure{TF_LT_CUBLAS_ERROR, "cuBLASLt status " + std::to_string(int(error))};
}
template<class T = void> T* pointer(uint64_t address) { return reinterpret_cast<T*>(uintptr_t(address)); }
struct Key {
    uint32_t mode, m, n, k, ldx, out, b_alignment, d_alignment;
    bool operator==(const Key& other) const {
        return mode == other.mode && m == other.m && n == other.n && k == other.k && ldx == other.ldx &&
            out == other.out && b_alignment == other.b_alignment && d_alignment == other.d_alignment;
    }
};
struct Plan {
    struct Candidate { cublasLtMatmulAlgo_t algorithm; size_t workspace; int32_t id; };
    Key key{};
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, out = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    size_t workspace = 0;
    std::array<Candidate, 16> eligible{};
    size_t eligible_count = 0, selected = 0;
    bool tuned = false;
    double tuned_ms = 0;
    ~Plan() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (out) cublasLtMatrixLayoutDestroy(out);
        if (desc) cublasLtMatmulDescDestroy(desc);
    }
    template<class T> void set(cublasLtMatmulDescAttributes_t attribute, const T& value) {
        lt(cublasLtMatmulDescSetAttribute(desc, attribute, &value, sizeof(value)));
    }
    template<class T> void prefer(cublasLtMatmulPreferenceAttributes_t attribute, const T& value) {
        lt(cublasLtMatmulPreferenceSetAttribute(preference, attribute, &value, sizeof(value)));
    }
    void init(cublasLtHandle_t handle, Key value, const tf_lt_create_v1& storage) {
        key = value;
        lt(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        const cublasOperation_t transposed = CUBLAS_OP_T, ordinary = CUBLAS_OP_N;
        set(CUBLASLT_MATMUL_DESC_TRANSA, transposed); set(CUBLASLT_MATMUL_DESC_TRANSB, ordinary);
        const int8_t no_fast_accum = 0; set(CUBLASLT_MATMUL_DESC_FAST_ACCUM, no_fast_accum);
        const bool narrow = key.mode == TF_LT_NVFP4, fp8 = key.mode >= TF_LT_FP8;
        if (narrow || fp8) {
            const int32_t mode = narrow ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 :
                CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
            set(CUBLASLT_MATMUL_DESC_A_SCALE_MODE, mode); set(CUBLASLT_MATMUL_DESC_B_SCALE_MODE, mode);
            const void* weight_scale = pointer(storage.weight_scales.address);
            const void* activation_scale = pointer(fp8 ? storage.weight_scales.address : storage.activation_scales.address);
            set(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, weight_scale);
            set(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, activation_scale);
        }
        const cudaDataType_t type = narrow ? CUDA_R_4F_E2M1 : fp8 ? CUDA_R_8F_E4M3 : CUDA_R_16BF;
        lt(cublasLtMatrixLayoutCreate(&a, type, key.k, key.n, key.k));
        lt(cublasLtMatrixLayoutCreate(&b, type, key.k, key.m, narrow || fp8 ? key.k : key.ldx));
        lt(cublasLtMatrixLayoutCreate(&out, fp8 || key.out == TF_LT_OUT_F32 ? CUDA_R_32F : CUDA_R_16BF, key.n, key.m, key.n));
        lt(cublasLtMatmulPreferenceCreate(&preference));
        prefer(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, storage.workspace.bytes);
        const uint32_t reductions = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
        prefer(CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, reductions);
        prefer(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, tf_lt_detail::alignment(storage.weight_raw.address));
        prefer(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, key.b_alignment);
        prefer(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, key.d_alignment);
        prefer(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, key.d_alignment);
        std::array<cublasLtMatmulHeuristicResult_t, 16> candidates{};
        int count = 0;
        const auto status = cublasLtMatmulAlgoGetHeuristic(handle, desc, a, b, out, out, preference,
            int(candidates.size()), candidates.data(), &count);
        if (status == CUBLAS_STATUS_NOT_SUPPORTED) throw Failure{TF_LT_UNSUPPORTED, "no cuBLASLt heuristic for this shape"};
        lt(status);
        for (const auto& candidate : candidates) {
            if (count-- <= 0) break;
            if (candidate.state != CUBLAS_STATUS_SUCCESS || candidate.workspaceSize > storage.workspace.bytes) continue;
            uint64_t flags = 0; uint32_t reduction = 0; size_t written = 0;
            lt(cublasLtMatmulAlgoCapGetAttribute(&candidate.algo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
                &flags, sizeof(flags), &written));
            lt(cublasLtMatmulAlgoConfigGetAttribute(&candidate.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                &reduction, sizeof(reduction), &written));
            if ((flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK) != CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F ||
                (reduction != CUBLASLT_REDUCTION_SCHEME_NONE && reduction != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)) continue;
            cublasLtMatmulHeuristicResult_t checked{};
            const auto check = cublasLtMatmulAlgoCheck(handle, desc, a, b, out, out, &candidate.algo, &checked);
            if (check == CUBLAS_STATUS_NOT_SUPPORTED) continue;
            lt(check);
            if (checked.state != CUBLAS_STATUS_SUCCESS || checked.workspaceSize > storage.workspace.bytes) continue;
            int32_t id = -1;
            lt(cublasLtMatmulAlgoConfigGetAttribute(&candidate.algo, CUBLASLT_ALGO_CONFIG_ID,
                &id, sizeof(id), &written));
            eligible[eligible_count++] = {candidate.algo, checked.workspaceSize, id};
        }
        if (!eligible_count) throw Failure{TF_LT_UNSUPPORTED, "no FP32-accumulator algorithm without output-type or in-place reduction"};
        algorithm = eligible[0].algorithm; workspace = eligible[0].workspace;
    }
};
} // namespace

struct tf_lt_handle {
    tf_lt_create_v1 storage{};
    cublasLtHandle_t library = nullptr;
    CUcontext context = nullptr;
    std::list<std::unique_ptr<Plan>> plans;
    std::mutex mutex;
    char error[512] = "";
    uint64_t last_stream = 0;
    bool submitted = false;
    tf_lt_run_v1 last_call{};
    Plan* last_plan = nullptr;
    ~tf_lt_handle() { plans.clear(); if (library) cublasLtDestroy(library); }
};

namespace {
tf_lt_status record(tf_lt_handle* handle, tf_lt_status status, const char* text) {
    std::snprintf(handle ? handle->error : creation_error, 512, "%s", text); return status;
}
tf_lt_status check(tf_lt_handle* h, tf_lt_detail::Check value) { return record(h, value.status, value.message); }
void current(tf_lt_handle* h) {
    CUcontext context = nullptr; driver(cuCtxGetCurrent(&context));
    if (!context || context != h->context) throw Failure{TF_LT_CONTEXT_MISMATCH, "the bridge's CUDA context must be current"};
}
bool ready(tf_lt_handle* h) {
    if (!h->submitted) return true;
    const auto status = cudaStreamQuery(pointer<CUstream_st>(h->last_stream));
    if (status == cudaErrorNotReady) return false;
    runtime(status); return true;
}
Plan& plan_for(tf_lt_handle* h, const tf_lt_run_v1& d) {
    const bool converted = d.mode != TF_LT_AFFINE4_BF16, fp8 = d.mode >= TF_LT_FP8;
    const Key key{d.mode, d.m, d.n, d.k, converted ? d.k : d.ldx, d.out_kind,
        tf_lt_detail::alignment(converted ? h->storage.activation_raw.address : d.x),
        tf_lt_detail::alignment(fp8 ? h->storage.activation_scales.address : d.out)};
    for (auto it = h->plans.begin(); it != h->plans.end(); ++it) if ((*it)->key == key) {
        h->plans.splice(h->plans.begin(), h->plans, it); return *h->plans.front();
    }
    const size_t limit = h->storage.max_plans ? h->storage.max_plans : 16;
    if (h->plans.size() == limit) throw Failure{TF_LT_UNSUPPORTED, "plan cache is full; no cached plan is evicted"};
    auto next = std::make_unique<Plan>(); next->init(h->library, key, h->storage);
    h->plans.push_front(std::move(next)); return *h->plans.front();
}
tf_lt_status exception(tf_lt_handle* handle) {
    try { throw; }
    catch (const Failure& f) { return record(handle, f.status, f.text.c_str()); }
    catch (const std::bad_alloc&) { return record(handle, TF_LT_OUT_OF_MEMORY, "CPU allocation failed"); }
    catch (const std::exception& e) { return record(handle, TF_LT_INTERNAL_ERROR, e.what()); }
    catch (...) { return record(handle, TF_LT_INTERNAL_ERROR, "unknown bridge exception"); }
}
} // namespace

extern "C" TF_LT_API uint32_t TF_LT_CALL tf_lt_abi_version(void) { return TF_LT_ABI_VERSION; }
extern "C" TF_LT_API tf_lt_status TF_LT_CALL tf_lt_get_requirements(const tf_lt_run_v1* d, tf_lt_requirements_v1* out) {
    try {
        if (!d || !out || out->abi_version != TF_LT_ABI_VERSION || out->struct_size != sizeof(*out))
            return record(nullptr, TF_LT_INVALID_ARGUMENT, "invalid requirements arguments");
        tf_lt_requirements_v1 required{};
        const auto status = tf_lt_detail::requirements(*d, required);
        if (!status) return check(nullptr, status);
        *out = required; return record(nullptr, TF_LT_OK, "");
    } catch (...) { return exception(nullptr); }
}
extern "C" TF_LT_API tf_lt_status TF_LT_CALL tf_lt_create(const tf_lt_create_v1* d, tf_lt_handle** out) {
    if (out) *out = nullptr;
    try {
        if (!d || !out) return record(nullptr, TF_LT_INVALID_ARGUMENT, "null create arguments");
        const auto valid = tf_lt_detail::create(*d); if (!valid) return check(nullptr, valid);
        auto h = std::make_unique<tf_lt_handle>(); h->storage = *d;
        driver(cuCtxGetCurrent(&h->context));
        if (!h->context) return record(nullptr, TF_LT_CONTEXT_MISMATCH, "create needs a current CUDA context");
        lt(cublasLtCreate(&h->library)); *out = h.release(); return record(nullptr, TF_LT_OK, "");
    } catch (...) { return exception(nullptr); }
}
extern "C" TF_LT_API tf_lt_status TF_LT_CALL tf_lt_run(tf_lt_handle* h, const tf_lt_run_v1* d) {
    if (!h) return record(nullptr, TF_LT_INVALID_ARGUMENT, "null run handle");
    try {
        std::lock_guard<std::mutex> lock(h->mutex);
        if (!d) return record(h, TF_LT_INVALID_ARGUMENT, "null run arguments");
        tf_lt_requirements_v1 sizes{};
        auto valid = tf_lt_detail::requirements(*d, sizes); if (!valid) return check(h, valid);
        valid = tf_lt_detail::buffers(h->storage, *d, sizes); if (!valid) return check(h, valid);
        current(h);
        if (h->submitted && h->last_stream != d->stream && !ready(h))
            return record(h, TF_LT_BUSY, "previous stream still uses the caller-owned scratch");
        Plan& plan = plan_for(h, *d); // Unsupported plans return before any output or GPU scratch write.
        h->last_plan = nullptr; // A preparation failure must not leave an older call eligible for tuning.
        h->last_stream = d->stream; h->submitted = true; // A later launch failure still leaves earlier work ordered.
        const auto stream = pointer<CUstream_st>(d->stream);
        const dim3 grid(d->k / 64, d->npad / 64);
        const void* activation = pointer(d->x);
        void* product = pointer(d->out);
        float alpha = 1;
        if (d->mode == TF_LT_AFFINE4_BF16) {
            tf_lt_gpu::affine4<<<grid, 256, 0, stream>>>(pointer<uint32_t>(d->weight), pointer<uint16_t>(d->scale),
                pointer<uint16_t>(d->bias), pointer<uint16_t>(h->storage.weight_raw.address), d->n, d->k, d->npad);
            runtime(cudaPeekAtLastError());
        } else if (d->mode == TF_LT_NVFP4) {
            runtime(cudaMemsetAsync(pointer(h->storage.weight_scales.address), 0, sizes.weight_scales_bytes, stream));
            runtime(cudaMemsetAsync(pointer(h->storage.activation_scales.address), 0, sizes.activation_scales_bytes, stream));
            tf_lt_gpu::nvfp4_unpack<<<grid, 256, 0, stream>>>(pointer<uint8_t>(d->weight), pointer<uint8_t>(d->scale),
                pointer<uint8_t>(h->storage.weight_raw.address), pointer<uint8_t>(h->storage.weight_scales.address), d->n, d->k);
            runtime(cudaPeekAtLastError());
            const size_t groups = size_t(d->m) * (d->k / 16);
            tf_lt_gpu::nvfp4_quant<<<unsigned((groups + 255) / 256), 256, 0, stream>>>(pointer<__nv_bfloat16>(d->x),
                pointer<uint8_t>(h->storage.activation_raw.address), pointer<uint8_t>(h->storage.activation_scales.address),
                d->m, d->k, d->ldx, 1.0f / d->act_scale);
            runtime(cudaPeekAtLastError());
            activation = pointer(h->storage.activation_raw.address);
            volatile float product = d->act_scale * d->weight_scale; alpha = product;
        } else {
            tf_lt_gpu::fp8_units<<<1, 32, 0, stream>>>(pointer<float>(h->storage.weight_scales.address));
            tf_lt_gpu::fp8_unpack<<<grid, 256, 0, stream>>>(pointer<uint8_t>(d->weight),
                pointer<uint8_t>(h->storage.weight_raw.address), d->n, d->k);
            const uint64_t row_scales = h->storage.activation_scales.address + uint64_t(d->m) * d->n * 4;
            tf_lt_gpu::fp8_rows<<<d->m, 256, 0, stream>>>(pointer<__nv_bfloat16>(d->x),
                pointer<uint8_t>(h->storage.activation_raw.address), pointer<float>(row_scales),
                d->m, d->k, d->ldx, d->mode == TF_LT_FP8_ROW, d->act_scale);
            runtime(cudaPeekAtLastError());
            product = pointer(h->storage.activation_scales.address);
            activation = pointer(h->storage.activation_raw.address);
        }
        const float zero = 0;
        lt(cublasLtMatmul(h->library, plan.desc, &alpha, pointer(h->storage.weight_raw.address), plan.a,
            activation, plan.b, &zero, product, plan.out, product, plan.out, &plan.algorithm,
            pointer(h->storage.workspace.address), h->storage.workspace.bytes, stream));
        if (d->mode >= TF_LT_FP8) {
            const size_t count = size_t(d->m) * d->n;
            const uint64_t row_scales = h->storage.activation_scales.address + count * 4;
            if (d->out_kind == TF_LT_OUT_F32)
                tf_lt_gpu::fp8_finish<true><<<unsigned((count + 255) / 256), 256, 0, stream>>>(
                    static_cast<const float*>(product), pointer<float>(row_scales), pointer<float>(d->scale),
                    pointer(d->out), count, d->n, d->k, d->mode == TF_LT_FP8_ROW, d->weight_scale);
            else tf_lt_gpu::fp8_finish<false><<<unsigned((count + 255) / 256), 256, 0, stream>>>(
                static_cast<const float*>(product), pointer<float>(row_scales), pointer<float>(d->scale),
                pointer(d->out), count, d->n, d->k, d->mode == TF_LT_FP8_ROW, d->weight_scale);
            runtime(cudaPeekAtLastError());
        }
        h->last_call = *d; h->last_plan = &plan;
        return record(h, TF_LT_OK, "");
    } catch (...) {
        try { std::lock_guard<std::mutex> lock(h->mutex); return exception(h); }
        catch (...) { return TF_LT_INTERNAL_ERROR; }
    }
}
extern "C" TF_LT_API tf_lt_status TF_LT_CALL tf_lt_tune_last(tf_lt_handle* h) {
    if (!h) return record(nullptr, TF_LT_INVALID_ARGUMENT, "null tuning handle");
    try {
        std::lock_guard<std::mutex> lock(h->mutex);
        current(h);
        if (!h->last_plan) return record(h, TF_LT_INVALID_ARGUMENT, "tuning requires a successful prepared run");
        const auto& call = h->last_call;
        Plan& plan = *h->last_plan;
        const auto stream = pointer<CUstream_st>(call.stream);
        cudaStreamCaptureStatus capture;
        runtime(cudaStreamIsCapturing(stream, &capture));
        if (capture != cudaStreamCaptureStatusNone) return record(h, TF_LT_UNSUPPORTED, "startup tuning is not allowed during capture");
        runtime(cudaStreamSynchronize(stream));
        if (!plan.tuned) {
            tf_lt_requirements_v1 required{};
            const auto valid = tf_lt_detail::requirements(call, required);
            if (!valid) return check(h, valid);
            tf_lt_detail::TuneStorage storage;
            if (!tf_lt_detail::tune_storage(h->storage, call, required, plan.key.d_alignment, storage))
                return record(h, TF_LT_UNSUPPORTED, "caller scratch has no temporary tuning output");
            struct Events {
                cudaEvent_t begin = nullptr, end = nullptr;
                ~Events() { if (end) cudaEventDestroy(end); if (begin) cudaEventDestroy(begin); }
            } events;
            runtime(cudaEventCreate(&events.begin)); runtime(cudaEventCreate(&events.end));
            float alpha = 1, zero = 0;
            if (call.mode == TF_LT_NVFP4) { volatile float product = call.act_scale * call.weight_scale; alpha = product; }
            const void* input = pointer(call.mode == TF_LT_AFFINE4_BF16 ? call.x : h->storage.activation_raw.address);
            double best_ms = std::numeric_limits<double>::infinity();
            size_t best = 0, timed = 0;
            for (size_t index = 0; index < plan.eligible_count; ++index) {
                const auto& candidate = plan.eligible[index];
                if (candidate.workspace > storage.workspace) continue;
                auto launch = [&] {
                    lt(cublasLtMatmul(h->library, plan.desc, &alpha, pointer(h->storage.weight_raw.address), plan.a,
                        input, plan.b, &zero, pointer(storage.output), plan.out, pointer(storage.output), plan.out,
                        &candidate.algorithm, pointer(h->storage.workspace.address), storage.workspace, stream));
                };
                for (int warm = 0; warm < 4; ++warm) launch();
                std::array<float, 3> times{};
                for (float& ms : times) {
                    runtime(cudaEventRecord(events.begin, stream));
                    for (int repeat = 0; repeat < 8; ++repeat) launch();
                    runtime(cudaEventRecord(events.end, stream)); runtime(cudaEventSynchronize(events.end));
                    runtime(cudaEventElapsedTime(&ms, events.begin, events.end)); ms /= 8;
                }
                std::sort(times.begin(), times.end()); ++timed;
                if (times[1] < best_ms) { best_ms = times[1]; best = index; }
            }
            if (!timed) return record(h, TF_LT_UNSUPPORTED, "no candidate fits workspace beside temporary tuning output");
            plan.selected = best; plan.algorithm = plan.eligible[best].algorithm;
            plan.workspace = plan.eligible[best].workspace; plan.tuned_ms = best_ms; plan.tuned = true;
        }
        std::snprintf(h->error, sizeof(h->error), "tuned candidate=%zu algorithm=%d eligible=%zu GEMM_ms=%.6f workspace=%zu",
            plan.selected, plan.eligible[plan.selected].id, plan.eligible_count, plan.tuned_ms, plan.workspace);
        return TF_LT_OK;
    } catch (...) {
        try { std::lock_guard<std::mutex> lock(h->mutex); return exception(h); }
        catch (...) { return TF_LT_INTERNAL_ERROR; }
    }
}
extern "C" TF_LT_API tf_lt_status TF_LT_CALL tf_lt_destroy(tf_lt_handle* h) {
    if (!h) return TF_LT_OK;
    try {
        { std::lock_guard<std::mutex> lock(h->mutex); current(h);
            if (!ready(h)) return record(h, TF_LT_BUSY, "synchronize the last stream before destroying the bridge");
            h->plans.clear(); lt(cublasLtDestroy(h->library)); h->library = nullptr;
        }
        delete h; return TF_LT_OK;
    } catch (...) { return exception(h); }
}
extern "C" TF_LT_API uint64_t TF_LT_CALL tf_lt_last_error(tf_lt_handle* h, char* destination, uint64_t capacity) {
    try {
        std::unique_lock<std::mutex> lock;
        if (h) lock = std::unique_lock<std::mutex>(h->mutex);
        const char* source = h ? h->error : creation_error;
        const size_t required = std::strlen(source) + 1;
        if (destination && capacity) {
            const size_t count = std::min<size_t>(required - 1, capacity - 1);
            std::memcpy(destination, source, count); destination[count] = 0;
        }
        return required;
    } catch (...) { if (destination && capacity) destination[0] = 0; return 1; }
}
