// The production address resolver preserves nulls and accepts shuffled backup planes without a legacy base.
#include "../../kernels/cuda/common/qwen_before.h"
#include <stdio.h>
#include <string.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "before resolver failed at line %d\n", __LINE__); return 1; } } while (0)

static int plant_null_offset;
static int plant_legacy_presence;

static uint64_t address(TfQ35BeforePlane p, size_t layer) {
    if (plant_null_offset && p.layers == NULL) return p.base + layer * p.stride;
    if (plant_legacy_presence && p.base == 0) return 0;
    return tf_q35_before_address(p, layer);
}

static int null_planes(void) {
    const TfQ35BeforePlane p = tf_q35_before_contiguous(0, 32);
    for (size_t layer = 0; layer < 7; ++layer) REQUIRE(address(p, layer) == 0);
    REQUIRE(tf_q35_before_tail(0, 4096) == 0);
    REQUIRE(tf_q35_before_tail(4096, 4096) == 8192);
    REQUIRE(tf_q35_before_address(tf_q35_before_table(NULL), 3) == 0);
    REQUIRE(0 + 3 * p.stride != 0);
    return 0;
}

static void backup(TfQ35BeforePlane p, size_t layer, const void* src, size_t bytes) {
    const uint64_t dst = address(p, layer);
    if (dst != 0) memcpy((void*)(uintptr_t)dst, src, bytes);
}

static int populated_planes(void) {
    enum { layers = 3, rec_bytes = 32, tail_bytes = 10, state_bytes = layers * rec_bytes, bytes = state_bytes + layers * tail_bytes };
    unsigned char kept[bytes + 16];
    unsigned char state[layers][rec_bytes], tails[layers][tail_bytes];
    memset(kept, 0xa5, sizeof kept);
    for (size_t layer = 0; layer < layers; ++layer) {
        for (size_t i = 0; i < rec_bytes; ++i) state[layer][i] = (unsigned char)(17 + layer * 51 + i * 7);
        for (size_t i = 0; i < tail_bytes; ++i) tails[layer][i] = (unsigned char)(29 + layer * 33 + i * 11);
    }
    const uint64_t base = (uint64_t)(uintptr_t)(kept + 8);
    const TfQ35BeforePlane rec = tf_q35_before_contiguous(base, rec_bytes);
    const TfQ35BeforePlane conv = tf_q35_before_contiguous(tf_q35_before_tail(base, state_bytes), tail_bytes);
    for (size_t layer = 0; layer < layers; ++layer) {
        REQUIRE(address(rec, layer) == base + layer * rec_bytes);
        REQUIRE(address(conv, layer) == base + state_bytes + layer * tail_bytes);
        backup(rec, layer, state[layer], rec_bytes);
        backup(conv, layer, tails[layer], tail_bytes);
    }
    REQUIRE(memcmp(kept + 8, state, sizeof state) == 0);
    REQUIRE(memcmp(kept + 8 + state_bytes, tails, sizeof tails) == 0);
    for (size_t i = 0; i < 8; ++i) REQUIRE(kept[i] == 0xa5 && kept[8 + bytes + i] == 0xa5);
    return 0;
}

static int table_planes(void) {
    unsigned char rec[3][48], conv[3][26];
    unsigned char state[32], tail[10];
    memset(rec, 0xa5, sizeof rec);
    memset(conv, 0xa5, sizeof conv);
    for (size_t i = 0; i < sizeof state; ++i) state[i] = (unsigned char)(i * 13 + 41);
    for (size_t i = 0; i < sizeof tail; ++i) tail[i] = (unsigned char)(i * 19 + 7);
    const uint64_t recurrence[3] = {(uint64_t)(uintptr_t)(rec[2] + 8), 0, (uint64_t)(uintptr_t)(rec[0] + 8)};
    const uint64_t convolution[3] = {(uint64_t)(uintptr_t)(conv[1] + 8), (uint64_t)(uintptr_t)(conv[2] + 8), 0};
    const TfQ35BeforePlane r = tf_q35_before_table(recurrence), c = tf_q35_before_table(convolution);
    REQUIRE(r.base == 0 && c.base == 0);
    for (size_t layer = 0; layer < 3; ++layer) {
        REQUIRE(address(r, layer) == recurrence[layer]);
        REQUIRE(address(c, layer) == convolution[layer]);
        backup(r, layer, state, sizeof state);
        backup(c, layer, tail, sizeof tail);
    }
    REQUIRE(memcmp(rec[2] + 8, state, sizeof state) == 0);
    REQUIRE(memcmp(rec[0] + 8, state, sizeof state) == 0);
    REQUIRE(memcmp(conv[1] + 8, tail, sizeof tail) == 0);
    REQUIRE(memcmp(conv[2] + 8, tail, sizeof tail) == 0);
    for (size_t i = 0; i < sizeof rec[1]; ++i) REQUIRE(rec[1][i] == 0xa5);
    for (size_t i = 0; i < sizeof conv[0]; ++i) REQUIRE(conv[0][i] == 0xa5);
    for (size_t layer = 0; layer < 3; ++layer) for (size_t i = 0; i < 8; ++i) {
        REQUIRE(rec[layer][i] == 0xa5 && rec[layer][40 + i] == 0xa5);
        REQUIRE(conv[layer][i] == 0xa5 && conv[layer][18 + i] == 0xa5);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 2) {
        plant_null_offset = strcmp(argv[1], "--plant-null-offset") == 0;
        plant_legacy_presence = strcmp(argv[1], "--plant-legacy-presence") == 0;
        if (!plant_null_offset && !plant_legacy_presence) return 2;
    } else if (argc != 1) return 2;
    if (null_planes() || populated_planes() || table_planes()) return 1;
    puts("PASS qwen before resolver: null, contiguous and shuffled table backups, legacy base zero, guards");
    return 0;
}
