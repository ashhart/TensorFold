#pragma once
#include <stdint.h>

typedef struct { const char *name; const uint32_t *logits_bits; const uint32_t *expected_bits; const uint32_t *history; uint32_t n, history_n, penalty_bits, expected_greedy; } HfRepetitionCase;
static const uint32_t hf_0_logits_bits[] = {0x00000000u, 0x80000000u, 0x40000000u, 0xc0000000u, 0x40400000u, 0xc0400000u};
static const uint32_t hf_0_expected_bits[] = {0x00000000u, 0x80000000u, 0x3fe8ba2eu, 0xc00ccccdu, 0x40400000u, 0xc0400000u};
static const uint32_t hf_0_history[] = {0u, 1u, 2u, 2u, 3u, 3u};
static const uint32_t hf_1_logits_bits[] = {0x3f800000u, 0xbf800000u, 0x00000000u, 0x80000000u};
static const uint32_t hf_1_expected_bits[] = {0x3f800000u, 0xbf800000u, 0x00000000u, 0x80000000u};
static const uint32_t hf_1_history[] = {0u, 1u, 2u, 3u};
static const uint32_t hf_2_logits_bits[] = {0x3f800000u, 0xbf800000u, 0x40000000u, 0xc0000000u};
static const uint32_t hf_2_expected_bits[] = {0x40000000u, 0xbf000000u, 0x40800000u, 0xbf800000u};
static const uint32_t hf_2_history[] = {0u, 1u, 2u, 3u};
static const uint32_t hf_3_logits_bits[] = {0x40800000u, 0x40400000u, 0x40000000u, 0x3f800000u, 0xbf800000u};
static const uint32_t hf_3_expected_bits[] = {0x4068ba2eu, 0x40400000u, 0x3fe8ba2eu, 0x3f68ba2eu, 0xbf8ccccdu};
static const uint32_t hf_3_history[] = {0u, 2u, 4u, 2u, 3u};
static const uint32_t hf_4_logits_bits[] = {0x3f800000u, 0x3f900000u, 0xbf800000u};
static const uint32_t hf_4_expected_bits[] = {0x3f800000u, 0x3f400000u, 0xbf800000u};
static const uint32_t hf_4_history[] = {1u};
static const uint32_t hf_5_logits_bits[] = {0x3f800000u, 0x3f8d0000u};
static const uint32_t hf_5_expected_bits[] = {0x3f800000u, 0x3f802e8bu};
static const uint32_t hf_5_history[] = {1u};
static const uint32_t hf_6_logits_bits[] = {0x7f800000u, 0xff800000u, 0x3f800000u, 0xbf800000u};
static const uint32_t hf_6_expected_bits[] = {0x7f800000u, 0xff800000u, 0x3f800000u, 0xbf800000u};
static const uint32_t hf_6_history[] = {0u, 1u};
static const uint32_t hf_7_logits_bits[] = {0x3f7f0000u, 0x3f810000u, 0xbf810000u, 0xbf7f0000u};
static const uint32_t hf_7_expected_bits[] = {0x3f67d174u, 0x3f6a8ba3u, 0xbf8de667u, 0xbf8c4000u};
static const uint32_t hf_7_history[] = {0u, 1u, 2u, 3u};
static const uint32_t hf_8_logits_bits[] = {0x3f800000u, 0xbf800000u};
static const uint32_t hf_8_expected_bits[] = {0x3f800000u, 0xbf800000u};
static const uint32_t hf_8_history[] = {0u};
static const uint32_t hf_9_logits_bits[] = {0x7fc00000u, 0x7fc00000u, 0x3f800000u, 0xbf800000u};
static const uint32_t hf_9_expected_bits[] = {0x7fc00000u, 0x7fc00000u, 0x3f68ba2eu, 0xbf800000u};
static const uint32_t hf_9_history[] = {0u, 2u};
static const HfRepetitionCase hf_repetition_cases[] = {
  {"sign_zero_duplicates", hf_0_logits_bits, hf_0_expected_bits, hf_0_history, 6u, 6u, 0x3f8ccccdu, 4u},
  {"identity", hf_1_logits_bits, hf_1_expected_bits, hf_1_history, 4u, 4u, 0x3f800000u, 0u},
  {"factor_below_one", hf_2_logits_bits, hf_2_expected_bits, hf_2_history, 4u, 4u, 0x3f000000u, 2u},
  {"prompt_and_committed", hf_3_logits_bits, hf_3_expected_bits, hf_3_history, 5u, 5u, 0x3f8ccccdu, 0u},
  {"greedy_flip", hf_4_logits_bits, hf_4_expected_bits, hf_4_history, 3u, 1u, 0x3fc00000u, 0u},
  {"bf16_rounding_trap", hf_5_logits_bits, hf_5_expected_bits, hf_5_history, 2u, 1u, 0x3f8ccccdu, 1u},
  {"infinities", hf_6_logits_bits, hf_6_expected_bits, hf_6_history, 4u, 2u, 0x3f8ccccdu, 0u},
  {"boundary_values", hf_7_logits_bits, hf_7_expected_bits, hf_7_history, 4u, 4u, 0x3f8ccccdu, 1u},
  {"empty_history", hf_8_logits_bits, hf_8_expected_bits, hf_8_history, 2u, 0u, 0x3f8ccccdu, 0u},
  {"nan_seen_unseen", hf_9_logits_bits, hf_9_expected_bits, hf_9_history, 4u, 2u, 0x3f8ccccdu, 0xffffffffu},
};
static const uint32_t hf_repetition_case_count = sizeof(hf_repetition_cases)/sizeof(hf_repetition_cases[0]);
