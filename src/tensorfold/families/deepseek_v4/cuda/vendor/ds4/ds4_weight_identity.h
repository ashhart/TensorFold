/* ds4_weight_identity.h — weight-file content fingerprint shared by the
 * ds4_weight_server manifest writer and the engine-side manifest import.
 *
 * Rider #48: the import used to validate identity by model_id + file size
 * only, so a weight server left running on a superseded checkpoint of
 * equal size fed stale bytes to a fresh engine silently (the golden-phase
 * mismatch signature reproduced digit-for-digit before the import was
 * even suspected).  The manifest now carries a content fingerprint per
 * model file and the importer refuses on mismatch.
 *
 * The fingerprint is a strided FNV-1a sample, not a full-file hash: the
 * head and tail windows are hashed completely (gguf magic, KV metadata
 * and the tensor index live in the head) plus one page per stride in
 * between.  Checkpoint mixups differ pervasively, so the sample catches
 * the incident class in milliseconds where a byte-serial FNV over an
 * 80 GiB file would cost the better part of a minute on every boot.
 * This is mixup detection, not tamper-proofing: a flipped byte between
 * sample windows is invisible by design.  The algo tag rides in the
 * manifest record so the geometry can change under a new tag without
 * breaking older readers (unknown record types are skipped by old
 * importers; unknown tags downgrade to a warning in new ones). */
#ifndef DS4_WEIGHT_IDENTITY_H
#define DS4_WEIGHT_IDENTITY_H

#include <stdint.h>

#define DS4_WFP_ALGO   "fnv1a-p16m-v1"
#define DS4_WFP_EDGE   (4ull << 20)  /* head and tail hashed fully */
#define DS4_WFP_STRIDE (16ull << 20) /* one page sampled per stride */
#define DS4_WFP_PAGE   4096ull

static inline uint64_t ds4_wfp_fnv1a(const void *p, uint64_t n, uint64_t h) {
    const unsigned char *b = (const unsigned char *)p;
    uint64_t i;
    for (i = 0; i < n; i++) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}

static inline uint64_t ds4_weight_content_fingerprint(const void *map,
                                                      uint64_t size) {
    const unsigned char *b = (const unsigned char *)map;
    uint64_t h = 1469598103934665603ull; /* FNV-1a offset basis */
    unsigned char sz[8];
    uint64_t off;
    int i;
    /* Fold the size so equal-prefix files of different length differ
     * even in the small-file full-hash branch. */
    for (i = 0; i < 8; i++) sz[i] = (unsigned char)(size >> (8 * i));
    h = ds4_wfp_fnv1a(sz, 8, h);
    if (!b || size == 0) return h;
    if (size <= 2 * DS4_WFP_EDGE) return ds4_wfp_fnv1a(b, size, h);
    h = ds4_wfp_fnv1a(b, DS4_WFP_EDGE, h);
    for (off = DS4_WFP_STRIDE; off + DS4_WFP_PAGE <= size - DS4_WFP_EDGE;
         off += DS4_WFP_STRIDE)
        h = ds4_wfp_fnv1a(b + off, DS4_WFP_PAGE, h);
    return ds4_wfp_fnv1a(b + (size - DS4_WFP_EDGE), DS4_WFP_EDGE, h);
}

#endif
