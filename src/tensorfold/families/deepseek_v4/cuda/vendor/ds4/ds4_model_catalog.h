#ifndef DS4_MODEL_CATALOG_H
#define DS4_MODEL_CATALOG_H

#include <stdint.h>

/* =========================================================================
 * memgov D1a-2: semantic tensor catalog (pure classification).
 *
 * A dependency-free leaf header in the ds4_mem_census.h convention: the
 * engine builds one catalog per model source at model_open, and the unit
 * suite drives the exact production classifier without a model file.
 *
 * The traits encode the engine's EXACT-NAME binding knowledge (weights_bind
 * / mtp_weights_bind / dspark_weights_bind suffixes) and the repack
 * candidacy rules (cuda/mmq/ds4_repack.cu ds4_repack_*_candidate), so the
 * residency planner can reason about tensor classes without substring
 * heuristics.  The legacy memmem(name, "_exps.") predicate at the HBM
 * pre-cache site survived ONE stage as a cross-check tripwire (zero
 * mismatches across the D1a gate batteries) and was retired in D1a-4b —
 * scoping sec 5: the heuristic died by cross-checked replacement; the
 * name relation stays pinned by the classifier units.
 *
 * Trait bits (0 = ALWAYS-HOT, the pre-cacheable default):
 * - ROUTED_EXPERT: the 3-D ffn_{gate,up,down}_exps stacks — top-K of N
 *   experts fire per token, so pre-caching them starves hot tensors.
 *   Deliberately matches the BINDER's names (blk.N./mtp.0./dspark.N.
 *   prefixes all share these suffixes); the 2-D shared-expert *_shexp
 *   tensors and exp_probs_b bias are NOT routed.
 * - ARTIFACT_REPLACED: an aligned repack artifact REPLACES the raw range
 *   when built/imported (IQ2_XXS gate/up, Q2_K down) — byte-neutral
 *   layouts, raw consumers fall back to the host mmap.
 * - ARTIFACT_ADDITIVE: an artifact may shadow the raw range for specific
 *   consumers while the raw stays served (Q8_0 aligned dense, Q8_0->f16
 *   colmajor prebuild).
 * - OPTIONAL: bound via the optional lookup (exp_probs_b.bias on the base
 *   model) — absence is legal.
 *
 * The type/dims/bytes gates on the artifact traits MIRROR the repack
 * candidate functions exactly; units pin both the suffix sets and the
 * mirror (a drifted repack rule fails the unit, not the field). */

enum {
    DS4_TCAT_ROUTED_EXPERT     = 1u << 0,
    DS4_TCAT_ARTIFACT_REPLACED = 1u << 1,
    DS4_TCAT_ARTIFACT_ADDITIVE = 1u << 2,
    DS4_TCAT_OPTIONAL          = 1u << 3
};

static inline int ds4_tcat_has_suffix(const char *name, uint64_t len,
                                      const char *sfx) {
    uint64_t sl = 0, i;
    if (!name || !sfx) return 0;
    while (sfx[sl]) sl++;
    if (len < sl) return 0;
    for (i = 0; i < sl; i++)
        if (name[len - sl + i] != sfx[i]) return 0;
    return 1;
}

static inline int ds4_tcat_contains(const char *name, uint64_t len,
                                    const char *needle) {
    uint64_t nl = 0, i, j;
    if (!name || !needle) return 0;
    while (needle[nl]) nl++;
    if (nl == 0 || len < nl) return 0;
    for (i = 0; i + nl <= len; i++) {
        for (j = 0; j < nl; j++)
            if (name[i + j] != needle[j]) break;
        if (j == nl) return 1;
    }
    return 0;
}

/* GGML type ids as the repack rules spell them (leaf header: no ggml
 * include, same literals-with-comments convention as ds4_repack.cu). */
#define DS4_TCAT_GGML_Q8_0    8u
#define DS4_TCAT_GGML_Q2_K    10u
#define DS4_TCAT_GGML_IQ2_XXS 16u

static inline uint32_t ds4_tensor_catalog_classify(const char *name,
                                                   uint64_t name_len,
                                                   uint32_t ndim,
                                                   uint32_t ggml_type,
                                                   const uint64_t *dims,
                                                   uint64_t bytes) {
    uint32_t traits = 0;
    const int gate = ds4_tcat_has_suffix(name, name_len, ".ffn_gate_exps.weight");
    const int up   = ds4_tcat_has_suffix(name, name_len, ".ffn_up_exps.weight");
    const int down = ds4_tcat_has_suffix(name, name_len, ".ffn_down_exps.weight");

    if (ndim == 3 && (gate || up || down)) {
        traits |= DS4_TCAT_ROUTED_EXPERT;
        /* ds4_repack_iq2_candidate mirror: IQ2_XXS gate/up stacks. */
        if (ggml_type == DS4_TCAT_GGML_IQ2_XXS && (gate || up) && dims &&
            dims[0] != 0 && dims[1] != 0 && dims[2] != 0 &&
            dims[2] <= UINT32_MAX && dims[0] % 1024u == 0 &&
            bytes != 0 && bytes % 66u == 0) {
            traits |= DS4_TCAT_ARTIFACT_REPLACED;
        }
        /* ds4_repack_q2k_candidate mirror: Q2_K down stacks. */
        if (ggml_type == DS4_TCAT_GGML_Q2_K && down && dims &&
            dims[0] != 0 && dims[1] != 0 && dims[2] != 0 &&
            dims[2] <= UINT32_MAX && dims[0] % 256u == 0 &&
            dims[1] % 2u == 0 && bytes != 0 && bytes % 84u == 0) {
            traits |= DS4_TCAT_ARTIFACT_REPLACED;
        }
    }

    if (ggml_type == DS4_TCAT_GGML_Q8_0 && ndim == 2 && dims &&
        dims[0] != 0 && dims[1] != 0 && bytes % 34u == 0 && bytes != 0) {
        /* ds4_repack_q8_candidate mirror: aligned dense (2 MiB floor,
         * token_embd excluded). */
        if (dims[0] % 1024u == 0 && bytes >= 2u * 1024u * 1024u &&
            !ds4_tcat_contains(name, name_len, "token_embd")) {
            traits |= DS4_TCAT_ARTIFACT_ADDITIVE;
        }
        /* ds4_repack_q8_f16_candidate mirror: f16 colmajor prebuild. */
        if (dims[0] % 32u == 0 &&
            ds4_tcat_contains(name, name_len, "attn_output_a.weight")) {
            traits |= DS4_TCAT_ARTIFACT_ADDITIVE;
        }
    }

    if (ds4_tcat_has_suffix(name, name_len, ".exp_probs_b.bias"))
        traits |= DS4_TCAT_OPTIONAL;

    return traits;
}

/* =========================================================================
 * memgov D1a-3: active intervals + canonical physical units (pure
 * compiler, plan §5.1/§5.2).
 *
 * The compiler turns a source's ACTIVE tensor set (full boot = every
 * tensor; slice boot = the slice's tensors ONLY — §5.1: a slice boot
 * must not enumerate the entire file) into canonical physical units:
 * coalesced source intervals with legal consumer boundaries, allocator
 * kind + rounding, and a residency-policy stamp with provenance in the
 * inputs.  D1a-3 is STRUCTURAL: the table is compiled and reconciled at
 * boot, but the live pre-cache/lookup paths still walk their current
 * structures; D1b's materializer adopts the table.
 *
 * Refinements over the live walk, encoded here and property-tested:
 * - a unit boundary NEVER splits a tensor (§5.2: chunking must not split
 *   a range a consumer later requests contiguously; the live walk chunks
 *   merged spans at arbitrary byte offsets).  A single tensor larger
 *   than max_unit_bytes becomes its own oversized unit.
 * - units are POLICY-HOMOGENEOUS: coalescing never merges across
 *   residency classes (the live walk gets this implicitly by excluding
 *   routed experts before merging; the compiler states it).
 * - import satisfaction is a pure rule: a unit is satisfied only by ONE
 *   device-resident import interval fully covering it; host-registered
 *   ranges and additive artifacts NEVER count as device coverage. */

enum {
    DS4_UPOL_DEVICE_PROMOTE = 0,     /* hot span: startup walk promotes  */
    DS4_UPOL_EXPERT_COLD,            /* routed experts, raw: stay on the
                                        mapped/mmap tier by design       */
    DS4_UPOL_ARTIFACT_REPLACED,      /* routed experts whose raw range a
                                        replace-kind artifact serves     */
    DS4_UPOL_HOST_MAPPED             /* whole-map registered tier (no
                                        per-span device promotion)       */
};

enum {
    DS4_UALLOC_NONE = 0,             /* no device backing planned        */
    DS4_UALLOC_VMM_ARENA,            /* 2 MiB-page VMM bump arena        */
    DS4_UALLOC_CUDAMALLOC            /* plain cudaMalloc span/arena      */
};

typedef struct {
    uint64_t off;                    /* absolute source offset           */
    uint64_t bytes;
    uint32_t traits;                 /* DS4_TCAT_* bits                  */
    uint8_t  active;                 /* member of the boot slice         */
} ds4_unit_tensor_in;

typedef struct {
    uint64_t src_off;                /* exact source interval            */
    uint64_t src_bytes;
    uint64_t planned_bytes;          /* after allocator rounding         */
    uint32_t first_tensor;           /* member range in the input array  */
    uint32_t n_tensors;              /* (the legal consumer boundaries)  */
    uint8_t  policy;                 /* DS4_UPOL_*                       */
    uint8_t  allocator;              /* DS4_UALLOC_*                     */
} ds4_phys_unit;

typedef struct {
    uint64_t merge_gap;              /* coalesce gap (live walk: 64 KiB) */
    uint64_t max_unit_bytes;         /* soft span cap (live walk knob)   */
    uint64_t vmm_granularity;        /* rounding for VMM units (0 = none)*/
    /* Effective-policy provenance (captured boot flags, plan §5.1): the
     * stamp is derived, the inputs say why.  memgov D3-2: the unit stamp
     * is TIMING-AGNOSTIC -- device_promote says whether non-expert units
     * ever device-promote (eager AND lazy residency both set it); the
     * eager-vs-lazy schedule lives on the SOURCE handle, not the unit. */
    uint8_t  device_promote;         /* non-expert units promote (source
                                        residency != HOST_MAPPED)        */
    uint8_t  replaces_complete;      /* every replace-candidate has an
                                        artifact (self-load / manifest)  */
} ds4_unit_compile_params;

static inline uint8_t ds4_unit_policy_of(uint32_t traits,
                                         const ds4_unit_compile_params *p) {
    if (traits & DS4_TCAT_ROUTED_EXPERT) {
        return (p->replaces_complete && (traits & DS4_TCAT_ARTIFACT_REPLACED))
            ? DS4_UPOL_ARTIFACT_REPLACED
            : DS4_UPOL_EXPERT_COLD;
    }
    return p->device_promote ? DS4_UPOL_DEVICE_PROMOTE : DS4_UPOL_HOST_MAPPED;
}

static inline uint64_t ds4_unit_round_planned(uint64_t bytes, uint8_t alloc,
                                              const ds4_unit_compile_params *p) {
    if (alloc == DS4_UALLOC_VMM_ARENA && p->vmm_granularity > 1) {
        const uint64_t g = p->vmm_granularity;
        return ((bytes + g - 1u) / g) * g;
    }
    return bytes;
}

/* Compile active tensors (SORTED by offset, non-overlapping) into units.
 * Returns the unit count, or -1 on invalid input (unsorted/overlapping
 * tensors, NULL outputs).  `units` must hold at least `n` entries (one
 * unit per tensor is the worst case: nothing ever merges). */
static inline int ds4_units_compile(const ds4_unit_tensor_in *ts, uint32_t n,
                                    const ds4_unit_compile_params *p,
                                    ds4_phys_unit *units) {
    int nu = 0;
    uint32_t i;
    uint64_t prev_end = 0;
    if (!p || (n != 0 && (!ts || !units))) return -1;
    for (i = 0; i < n; i++) {
        if (ts[i].off < prev_end) return -1;      /* unsorted or overlap */
        prev_end = ts[i].off + ts[i].bytes;
        if (!ts[i].active || ts[i].bytes == 0) continue;

        const uint8_t pol = ds4_unit_policy_of(ts[i].traits, p);
        const uint8_t alloc =
            (pol == DS4_UPOL_DEVICE_PROMOTE)
                ? (p->vmm_granularity > 1 ? DS4_UALLOC_VMM_ARENA
                                          : DS4_UALLOC_CUDAMALLOC)
                : DS4_UALLOC_NONE;
        ds4_phys_unit *u = nu > 0 ? &units[nu - 1] : 0;
        const int mergeable = u != 0 &&
            u->policy == pol && u->allocator == alloc &&
            ts[i].off >= u->src_off + u->src_bytes &&
            ts[i].off - (u->src_off + u->src_bytes) <= p->merge_gap &&
            (p->max_unit_bytes == 0 ||
             ts[i].off + ts[i].bytes - u->src_off <= p->max_unit_bytes) &&
            u->first_tensor + u->n_tensors == i;    /* contiguous members */
        if (mergeable) {
            u->src_bytes = ts[i].off + ts[i].bytes - u->src_off;
            u->n_tensors++;
        } else {
            u = &units[nu++];
            u->src_off = ts[i].off;
            u->src_bytes = ts[i].bytes;
            u->first_tensor = i;
            u->n_tensors = 1;
            u->policy = pol;
            u->allocator = alloc;
        }
        u->planned_bytes = ds4_unit_round_planned(u->src_bytes, u->allocator, p);
    }
    return nu;
}

/* Invariant verification (the compiler's own gate; the boot build runs it
 * and any violation counts a census fault).  Returns the violation count:
 * exact coverage of every active tensor, no unit overlap, ordering, no
 * tensor split across a unit boundary, policy homogeneity, rounding
 * exactness. */
static inline uint32_t ds4_units_verify(const ds4_unit_tensor_in *ts, uint32_t n,
                                        const ds4_unit_compile_params *p,
                                        const ds4_phys_unit *units, int nu) {
    uint32_t faults = 0;
    int ui;
    uint64_t prev_unit_end = 0;
    uint32_t next_member = 0;
    if (nu < 0 || !p) return 1;
    for (ui = 0; ui < nu; ui++) {
        const ds4_phys_unit *u = &units[ui];
        uint32_t k;
        if (u->src_off < prev_unit_end) faults++;          /* overlap/order */
        prev_unit_end = u->src_off + u->src_bytes;
        if (u->n_tensors == 0) faults++;
        for (k = 0; k < u->n_tensors; k++) {
            const uint32_t ti = u->first_tensor + k;
            if (ti >= n) { faults++; break; }
            const ds4_unit_tensor_in *t = &ts[ti];
            if (!t->active) faults++;
            /* no split: the whole tensor lies inside the unit */
            if (t->off < u->src_off ||
                t->off + t->bytes > u->src_off + u->src_bytes) faults++;
            /* policy homogeneity */
            if (ds4_unit_policy_of(t->traits, p) != u->policy) faults++;
        }
        if (u->planned_bytes !=
            ds4_unit_round_planned(u->src_bytes, u->allocator, p)) faults++;
        /* member indices advance without gaps over the active set */
        while (next_member < u->first_tensor) {
            if (next_member < n &&
                ts[next_member].active && ts[next_member].bytes != 0)
                faults++;                       /* active tensor uncovered */
            next_member++;
        }
        next_member = u->first_tensor + u->n_tensors;
    }
    while (next_member < n) {
        if (ts[next_member].active && ts[next_member].bytes != 0) faults++;
        next_member++;
    }
    return faults;
}

/* Import satisfaction (plan §5.2): satisfied iff ONE device-resident
 * import interval fully covers the unit.  Host-registered ranges and
 * additive artifacts are not device coverage BY RULE — callers must not
 * even pass them in. */
typedef struct {
    uint64_t off, bytes;
} ds4_unit_import_iv;

static inline int ds4_unit_import_satisfied(const ds4_phys_unit *u,
                                            const ds4_unit_import_iv *ivs,
                                            uint32_t n_ivs) {
    uint32_t i;
    if (!u) return 0;
    for (i = 0; i < n_ivs; i++) {
        if (ivs[i].off <= u->src_off &&
            ivs[i].off + ivs[i].bytes >= u->src_off + u->src_bytes)
            return 1;
    }
    return 0;
}

/* D1a-4: the unit span of a published physical range -- the publication
 * funnel's stamp.  Units are sorted and non-overlapping (ds4_units_verify),
 * so the units intersecting [off, off+bytes) form a contiguous run:
 * binary-search the first unit ending past `off`, then walk while units
 * start before the interval end.  Outputs are inclusive unit indices;
 * -1/-1 = no table, empty interval, or no intersection. */
static inline void ds4_units_span_of(const ds4_phys_unit *units, uint32_t n,
                                     uint64_t off, uint64_t bytes,
                                     int *lo, int *hi) {
    uint32_t a = 0, b = n, j;
    *lo = *hi = -1;
    if (!units || n == 0 || bytes == 0 || off + bytes < off) return;
    while (a < b) {
        const uint32_t mid = a + (b - a) / 2u;
        if (units[mid].src_off + units[mid].src_bytes <= off) a = mid + 1;
        else b = mid;
    }
    if (a >= n || units[a].src_off >= off + bytes) return;
    *lo = (int)a;
    j = a;
    while (j + 1 < n && units[j + 1].src_off < off + bytes) j++;
    *hi = (int)j;
}

/* Unit-table census for the boot line + gate reconcile. */
typedef struct {
    uint64_t units;
    uint64_t covered_bytes;          /* Σ src_bytes                      */
    uint64_t planned_bytes;          /* Σ planned (rounding included)    */
    uint64_t promote_bytes;          /* DEVICE_PROMOTE units             */
    uint64_t cold_bytes;             /* EXPERT_COLD + ARTIFACT_REPLACED  */
} ds4_unit_table_counts;

static inline void ds4_units_count(const ds4_phys_unit *units, int nu,
                                   ds4_unit_table_counts *out) {
    int i;
    if (!out) return;
    out->units = nu > 0 ? (uint64_t)nu : 0;
    out->covered_bytes = out->planned_bytes = 0;
    out->promote_bytes = out->cold_bytes = 0;
    for (i = 0; i < nu; i++) {
        out->covered_bytes += units[i].src_bytes;
        out->planned_bytes += units[i].planned_bytes;
        if (units[i].policy == DS4_UPOL_DEVICE_PROMOTE)
            out->promote_bytes += units[i].src_bytes;
        else if (units[i].policy == DS4_UPOL_EXPERT_COLD ||
                 units[i].policy == DS4_UPOL_ARTIFACT_REPLACED)
            out->cold_bytes += units[i].src_bytes;
    }
}

/* Per-catalog census: how many tensors carry each trait (the boot line
 * and the D1a gate's positive engagement signal render from this). */
typedef struct {
    uint64_t tensors;
    uint64_t routed;
    uint64_t replaced;
    uint64_t additive;
    uint64_t optional;
} ds4_model_catalog_counts;

static inline void ds4_model_catalog_count(const uint8_t *traits, uint64_t n,
                                           ds4_model_catalog_counts *out) {
    uint64_t i;
    if (!out) return;
    out->tensors = n;
    out->routed = out->replaced = out->additive = out->optional = 0;
    if (!traits) return;
    for (i = 0; i < n; i++) {
        if (traits[i] & DS4_TCAT_ROUTED_EXPERT)     out->routed++;
        if (traits[i] & DS4_TCAT_ARTIFACT_REPLACED) out->replaced++;
        if (traits[i] & DS4_TCAT_ARTIFACT_ADDITIVE) out->additive++;
        if (traits[i] & DS4_TCAT_OPTIONAL)          out->optional++;
    }
}

#endif /* DS4_MODEL_CATALOG_H */
