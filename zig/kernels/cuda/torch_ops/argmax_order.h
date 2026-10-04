#ifndef TENSORFOLD_ARGMAX_ORDER_H
#define TENSORFOLD_ARGMAX_ORDER_H

#include <stdint.h>
#include <stdbool.h>
#include <limits.h>

#ifdef __CUDACC__
#define TF_COMPARE static __host__ __device__ __forceinline__
#else
#define TF_COMPARE static inline
#endif

typedef struct {
    uint32_t key;
    uint32_t nan;
    uint64_t column;
} TFMaxCandidate;

TF_COMPARE TFMaxCandidate tf_max_empty(void) {
    TFMaxCandidate result = {0u, 0u, UINT64_MAX};
    return result;
}

TF_COMPARE TFMaxCandidate tf_max_bits(uint32_t word, uint64_t column) {
    const uint32_t magnitude = word & 0x7fffffffu;
    if (magnitude == 0u) word = 0u;
    TFMaxCandidate result;
    result.key = (word & 0x80000000u) ? ~word : (word ^ 0x80000000u);
    result.nan = magnitude > 0x7f800000u;
    result.column = column;
    return result;
}

TF_COMPARE bool tf_max_left_wins(uint64_t left_key, uint32_t left_nan, uint64_t left_column,
                                 uint64_t right_key, uint32_t right_nan, uint64_t right_column) {
    if (left_column == UINT64_MAX) return false;
    if (right_column == UINT64_MAX) return true;
    if (left_nan != right_nan) return left_nan != 0;
    if (!left_nan && left_key != right_key) return left_key > right_key;
#ifdef TF_ARGMAX_LAST_TIE
    return left_column > right_column;
#else
    return left_column < right_column;
#endif
}

TF_COMPARE TFMaxCandidate tf_max_pick(TFMaxCandidate left, TFMaxCandidate right) {
    return tf_max_left_wins(left.key, left.nan, left.column, right.key, right.nan, right.column) ? left : right;
}

TF_COMPARE bool tf_argmax_dimensions(int64_t rows, int64_t vocab, int64_t stride, int32_t dtype) {
    if (rows < 0 || rows > INT32_MAX || vocab < 1 || stride < vocab || (dtype != 0 && dtype != 1)) return false;
    if (rows == 0) return true;
    if (rows - 1 > (INT64_MAX - vocab) / stride) return false;
    return true;
}

TF_COMPARE bool tf_argmax_disjoint(uintptr_t source, uintptr_t destination, uint64_t source_bytes, uint64_t destination_bytes) {
    if (source > UINTPTR_MAX - source_bytes || destination > UINTPTR_MAX - destination_bytes) return false;
    return source + source_bytes <= destination || destination + destination_bytes <= source;
}

#undef TF_COMPARE
#endif
