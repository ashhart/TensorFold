// Versioned C ABI for an opt-in prompt projection experiment; GPU allocations remain caller-owned.
#ifndef TF_CUBLASLT_BRIDGE_H
#define TF_CUBLASLT_BRIDGE_H
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define TF_LT_CALL __cdecl
#if defined(TF_LT_BUILD)
#define TF_LT_API __declspec(dllexport)
#else
#define TF_LT_API __declspec(dllimport)
#endif
#else
#define TF_LT_CALL
#define TF_LT_API __attribute__((visibility("default")))
#endif

#define TF_LT_ABI_VERSION 1u
typedef uint32_t tf_lt_status;
enum {
    TF_LT_OK = 0, TF_LT_UNSUPPORTED = 1, TF_LT_INVALID_ARGUMENT = 2, TF_LT_BUFFER_TOO_SMALL = 3,
    TF_LT_CUDA_ERROR = 4, TF_LT_CUBLAS_ERROR = 5, TF_LT_CONTEXT_MISMATCH = 6,
    TF_LT_OUT_OF_MEMORY = 7, TF_LT_BUSY = 8, TF_LT_INTERNAL_ERROR = 9
};
enum { TF_LT_AFFINE4_BF16 = 1, TF_LT_NVFP4 = 2, TF_LT_FP8 = 3, TF_LT_FP8_ROW = 4 };
enum { TF_LT_OUT_BF16 = 1, TF_LT_OUT_F32 = 2 };
typedef struct tf_lt_handle tf_lt_handle;
typedef struct tf_lt_buffer_v1 { uint64_t address, bytes; } tf_lt_buffer_v1;

typedef struct tf_lt_create_v1 {
    uint32_t abi_version, struct_size, max_plans, flags;
    tf_lt_buffer_v1 weight_raw, activation_raw, weight_scales, activation_scales, workspace;
    uint64_t reserved[4];
} tf_lt_create_v1;

typedef struct tf_lt_run_v1 {
    uint32_t abi_version, struct_size, mode, out_kind;
    uint32_t m, n, k, npad, ldx, flags;
    uint64_t x, weight, scale, bias, out;
    float act_scale, weight_scale;
    uint64_t x_bytes, weight_bytes, scale_bytes, bias_bytes, out_bytes;
    uint64_t stream, reserved[3];
} tf_lt_run_v1;

typedef struct tf_lt_requirements_v1 {
    uint32_t abi_version, struct_size;
    uint64_t weight_raw_bytes, activation_raw_bytes, weight_scales_bytes, activation_scales_bytes;
    uint64_t minimum_workspace_bytes, preferred_workspace_bytes, reserved;
} tf_lt_requirements_v1;

#if defined(__cplusplus)
#define TF_LT_ASSERT static_assert
extern "C" {
#else
#define TF_LT_ASSERT _Static_assert
#endif
TF_LT_ASSERT(sizeof(tf_lt_buffer_v1) == 16, "buffer ABI");
TF_LT_ASSERT(sizeof(tf_lt_create_v1) == 128, "create ABI");
TF_LT_ASSERT(offsetof(tf_lt_create_v1, weight_raw) == 16, "create weight ABI");
TF_LT_ASSERT(offsetof(tf_lt_create_v1, workspace) == 80, "create workspace ABI");
TF_LT_ASSERT(offsetof(tf_lt_create_v1, reserved) == 96, "create reserved ABI");
TF_LT_ASSERT(sizeof(tf_lt_run_v1) == 160, "run ABI");
TF_LT_ASSERT(offsetof(tf_lt_run_v1, m) == 16, "run shape ABI");
TF_LT_ASSERT(offsetof(tf_lt_run_v1, x) == 40, "run pointer ABI");
TF_LT_ASSERT(offsetof(tf_lt_run_v1, act_scale) == 80, "run scale ABI");
TF_LT_ASSERT(offsetof(tf_lt_run_v1, x_bytes) == 88, "run size ABI");
TF_LT_ASSERT(offsetof(tf_lt_run_v1, stream) == 128, "run stream ABI");
TF_LT_ASSERT(offsetof(tf_lt_run_v1, reserved) == 136, "run reserved ABI");
TF_LT_ASSERT(sizeof(tf_lt_requirements_v1) == 64, "requirements ABI");

TF_LT_API uint32_t TF_LT_CALL tf_lt_abi_version(void);
TF_LT_API tf_lt_status TF_LT_CALL tf_lt_get_requirements(const tf_lt_run_v1*, tf_lt_requirements_v1*);
TF_LT_API tf_lt_status TF_LT_CALL tf_lt_create(const tf_lt_create_v1*, tf_lt_handle**);
TF_LT_API tf_lt_status TF_LT_CALL tf_lt_destroy(tf_lt_handle*);
TF_LT_API tf_lt_status TF_LT_CALL tf_lt_run(tf_lt_handle*, const tf_lt_run_v1*);
// Optional startup-only tuning after a successful run; synchronizes that stream and never writes the user output.
TF_LT_API tf_lt_status TF_LT_CALL tf_lt_tune_last(tf_lt_handle*);
// Copies a NUL-terminated diagnostic and returns required bytes including NUL; NULL handle gives this thread's error.
TF_LT_API uint64_t TF_LT_CALL tf_lt_last_error(tf_lt_handle*, char* destination, uint64_t capacity);

#if defined(__cplusplus)
}
#endif
#undef TF_LT_ASSERT
#endif
