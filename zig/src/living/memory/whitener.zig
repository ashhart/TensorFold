//! Dual-layer background whitener and high-specificity prototype selector.
//! Strips shared syntax/template principal components using only the model's own hidden states
//! (zero external embedding model) and scores token specificity against a generic calibration bank.
const std = @import("std");
const quant = @import("quant.zig");
const Allocator = std.mem.Allocator;

/// Generic calibration bank strictly disjoint from TensorFold's 150-prompt slide_panel.zig.
/// Covers declarative prose, interrogatives, generic personal templates, and generic fictional entities.
pub const calibration_texts = [_][]const u8{
    // Declarative general prose & common physical/engineering vocabulary
    "The morning train arrived at the central station five minutes ahead of schedule.",
    "Careful preparation of the soil helps seedlings establish deep roots during spring.",
    "Software engineers review pull requests and version control commits before release.",
    "A balanced meal includes proteins, complex carbohydrates, and fresh vegetables.",
    "Public transit systems reduce traffic congestion in densely populated cities.",
    "Acoustic guitars produce sound through the vibration of strings over a hollow body.",
    "Historical archives preserve letters, maps, and municipal records for future scholars.",
    "Regular maintenance extends the service life of industrial pumps, valves, and hydraulic compressors.",
    "Coastal towns often rely on fishing, shipping, and seasonal tourism.",
    "Clear documentation allows new team members to understand complex workflows quickly.",
    "Winter storms can disrupt power lines, wind turbines, and regional freight deliveries.",
    "Mathematicians use proofs to establish results with complete logical certainty.",
    "Modern hospitals coordinate care across emergency, surgical, and outpatient departments.",
    "Ceramic pottery is shaped on a wheel and hardened inside a high-temperature kiln.",
    "National parks protect native wildlife habitats and scenic geological formations.",
    "Mechanical locks, metal gears, springs, and pressure gauges require precise calibration.",
    "Weather stations track air pressure, humidity, wind direction, dry air, ice, cold fronts, and shadows.",
    // Generic interrogatives
    "What is the capital city of a European country?",
    "What is the purpose of a standard safety inspection?",
    "What is the function of an optical sensor in a factory?",
    "What is the difference between speed and velocity?",
    "What distinguishes an acid from a base in chemistry?",
    "Why does copper conduct heat better than wood?",
    "Why do seasons change throughout the calendar year?",
    "How does an internal combustion engine ignite fuel?",
    "How do satellites maintain a stable orbit around Earth?",
    "Who invented the modern printing press in Europe?",
    "When did the first transatlantic cable begin operation?",
    "Where is the largest freshwater basin on the continent?",
    "Which material resists corrosion best in marine environments?",
    // Generic personal-question templates (disjoint from CONTROLS_15)
    "What password did I set for my personal account?",
    "What song did I save to my personal playlist?",
    "What restaurant did I pick for my personal dinner?",
    "What movie did I choose for my personal watchlist?",
    "What address did I enter for my personal delivery?",
    "Do you remember what book I borrowed last week?",
    "Which sport do I enjoy watching on weekends?",
    // Generic fictional-entity templates (disjoint from NEIGHBOURS_20)
    "What is the height of the fictional Greystone lighthouse?",
    "What is the population of the fictional Meadowbrook village?",
    "What is the length of the fictional Copperhill canal?",
    "What is the opening date of the fictional Westgate market?",
    "What is the capacity of the fictional Northbay stadium?",
    "What is the ticket price of the fictional Riverbend cinema?",
    "What is the route of the fictional Silverbrook bus?",
    "What is the serial code of the fictional Ironbridge machine?",
};

pub const Transformed = struct {
    unit: []f32, // [n, dim] unit-normalized whitened vectors
    norms: []f32, // [n] post-whitening norms
    spec: []f32, // [n] specificity in [0, 1]

    pub fn deinit(self: *Transformed, gpa: Allocator) void {
        gpa.free(self.unit);
        gpa.free(self.norms);
        gpa.free(self.spec);
        self.* = undefined;
    }
};

pub const Prototypes = struct {
    vecs: []f32, // [count, dim]
    weights: []f32, // [count]
    count: usize,
    dim: usize,

    pub fn deinit(self: *Prototypes, gpa: Allocator) void {
        gpa.free(self.vecs);
        gpa.free(self.weights);
        self.* = undefined;
    }
};

pub const LayerWhitener = struct {
    dim: usize,
    k_pcs: usize,
    n_cal: usize,
    mean: []f32, // [dim]
    basis: []f32, // [k_pcs, dim]
    cal_tokens: []f32, // [n_cal, dim]
    norm_floor: f32,

    pub fn deinit(self: *LayerWhitener, gpa: Allocator) void {
        gpa.free(self.mean);
        gpa.free(self.basis);
        gpa.free(self.cal_tokens);
        self.* = undefined;
    }

    /// Fit mean, top-K background PCs, 35th-percentile norm floor, and deduplicated background token bank
    /// from concatenated calibration hidden states `raw [n_total, dim]`.
    pub fn fit(
        gpa: Allocator,
        raw: []const f32,
        n_total: usize,
        dim: usize,
        k_pcs_target: usize,
        max_cal_tokens: usize,
    ) !LayerWhitener {
        std.debug.assert(raw.len == n_total * dim and n_total > 0 and dim > 0);
        const mean = try gpa.alloc(f32, dim);
        errdefer gpa.free(mean);
        @memset(mean, 0);
        for (0..n_total) |r| {
            const row = raw[r * dim ..][0..dim];
            for (row, mean) |v, *m| m.* += v;
        }
        const inv_n: f32 = 1.0 / @as(f32, @floatFromInt(n_total));
        for (mean) |*m| m.* *= inv_n;

        const centered = try gpa.alloc(f32, n_total * dim);
        defer gpa.free(centered);
        for (0..n_total) |r| {
            const src = raw[r * dim ..][0..dim];
            const dst = centered[r * dim ..][0..dim];
            for (src, mean, dst) |v, m, *o| o.* = v - m;
        }

        const k_req = @min(k_pcs_target, @min(n_total, dim));
        const basis_buf = try gpa.alloc(f32, k_req * dim);
        errdefer gpa.free(basis_buf);
        const k_got = try quant.topRightSingularVectors(gpa, centered, n_total, dim, k_req, basis_buf);
        const basis = if (k_got == k_req) basis_buf else blk: {
            const shrunk = try gpa.dupe(f32, basis_buf[0 .. k_got * dim]);
            gpa.free(basis_buf);
            break :blk shrunk;
        };
        errdefer gpa.free(basis);

        // Project out top-K PCs in-place on `centered` -> `hw`
        projectOutBasis(centered, n_total, dim, basis, k_got);

        const norms = try gpa.alloc(f32, n_total);
        defer gpa.free(norms);
        for (0..n_total) |r| {
            const row = centered[r * dim ..][0..dim];
            const nrm = @sqrt(quant.dotF32(row, row));
            norms[r] = nrm;
            const inv = 1.0 / @max(nrm, 1e-8);
            for (row) |*v| v.* *= inv;
        }

        // 35th percentile norm floor
        const sorted_norms = try gpa.dupe(f32, norms);
        defer gpa.free(sorted_norms);
        std.mem.sort(f32, sorted_norms, {}, std.sort.asc(f32));
        const p35_idx = @min(sorted_norms.len - 1, (sorted_norms.len * 35) / 100);
        const norm_floor = sorted_norms[p35_idx];

        // Deduplicate calibration unit vectors by descending norm
        const cal_tokens = try dedupVectors(gpa, centered, norms, n_total, dim, max_cal_tokens, 0.92);
        return .{
            .dim = dim,
            .k_pcs = k_got,
            .n_cal = cal_tokens.len / dim,
            .mean = mean,
            .basis = basis,
            .cal_tokens = cal_tokens,
            .norm_floor = norm_floor,
        };
    }

    pub fn transform(self: *const LayerWhitener, gpa: Allocator, raw: []const f32, n: usize) !Transformed {
        std.debug.assert(raw.len == n * self.dim);
        const unit = try gpa.alloc(f32, n * self.dim);
        errdefer gpa.free(unit);
        const norms = try gpa.alloc(f32, n);
        errdefer gpa.free(norms);
        const spec = try gpa.alloc(f32, n);
        errdefer gpa.free(spec);

        for (0..n) |r| {
            const src = raw[r * self.dim ..][0..self.dim];
            const dst = unit[r * self.dim ..][0..self.dim];
            for (src, self.mean, dst) |v, m, *o| o.* = v - m;
        }
        projectOutBasis(unit, n, self.dim, self.basis, self.k_pcs);

        for (0..n) |r| {
            const row = unit[r * self.dim ..][0..self.dim];
            const nrm = @sqrt(quant.dotF32(row, row));
            norms[r] = nrm;
            const inv = 1.0 / @max(nrm, 1e-8);
            for (row) |*v| v.* *= inv;

            var max_bg: f32 = -1.0;
            for (0..self.n_cal) |j| {
                const bg = self.cal_tokens[j * self.dim ..][0..self.dim];
                const c = quant.dotF32(row, bg);
                if (c > max_bg) max_bg = c;
            }
            spec[r] = std.math.clamp(1.0 - max_bg, 0.0, 1.0);
        }
        return .{ .unit = unit, .norms = norms, .spec = spec };
    }
};

fn projectOutBasis(mat: []f32, rows: usize, dim: usize, basis: []const f32, k: usize) void {
    const V = @Vector(8, f32);
    for (0..rows) |r| {
        const row = mat[r * dim ..][0..dim];
        for (0..k) |p| {
            const bp = basis[p * dim ..][0..dim];
            const coeff = quant.dotF32(row, bp);
            if (coeff == 0.0) continue;
            const vc: V = @splat(coeff);
            var j: usize = 0;
            while (j + 8 <= dim) : (j += 8) {
                const rv: V = row[j..][0..8].*;
                const bv: V = bp[j..][0..8].*;
                row[j..][0..8].* = rv - vc * bv;
            }
            while (j < dim) : (j += 1) row[j] -= coeff * bp[j];
        }
    }
}

fn dedupVectors(
    gpa: Allocator,
    unit_h: []const f32,
    priorities: []const f32,
    n: usize,
    dim: usize,
    max_vecs: usize,
    dedup_cos: f32,
) ![]f32 {
    const order = try gpa.alloc(usize, n);
    defer gpa.free(order);
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order, priorities, struct {
        fn less(p: []const f32, a: usize, b: usize) bool {
            return p[a] > p[b];
        }
    }.less);

    var chosen: std.ArrayList(f32) = .empty;
    errdefer chosen.deinit(gpa);
    var count: usize = 0;
    for (order) |idx| {
        if (count >= max_vecs) break;
        const v = unit_h[idx * dim ..][0..dim];
        var dup = false;
        for (0..count) |j| {
            if (quant.dotF32(chosen.items[j * dim ..][0..dim], v) > dedup_cos) {
                dup = true;
                break;
            }
        }
        if (dup) continue;
        try chosen.appendSlice(gpa, v);
        count += 1;
    }
    if (count == 0 and n > 0) {
        try chosen.appendSlice(gpa, unit_h[order[0] * dim ..][0..dim]);
    }
    return chosen.toOwnedSlice(gpa);
}

/// Selects high-specificity, deduplicated prototype vectors and their specificity weights.
pub fn selectPrototypes(
    gpa: Allocator,
    tr: *const Transformed,
    dim: usize,
    norm_floor: f32,
    max_protos: usize,
    min_spec: f32,
    dedup_cos: f32,
) !Prototypes {
    const n = tr.norms.len;
    std.debug.assert(n > 0 and tr.unit.len == n * dim);

    const score = try gpa.alloc(f32, n);
    defer gpa.free(score);
    const order = try gpa.alloc(usize, n);
    defer gpa.free(order);
    for (0..n) |i| {
        score[i] = tr.norms[i] * std.math.pow(f32, tr.spec[i], 1.4);
        order[i] = i;
    }
    std.mem.sort(usize, order, score, struct {
        fn less(s: []const f32, a: usize, b: usize) bool {
            return s[a] > s[b];
        }
    }.less);

    var vecs: std.ArrayList(f32) = .empty;
    errdefer vecs.deinit(gpa);
    var weights: std.ArrayList(f32) = .empty;
    errdefer weights.deinit(gpa);

    for (order) |idx| {
        if (weights.items.len >= max_protos) break;
        if (tr.spec[idx] < min_spec and weights.items.len >= 6) continue;
        if (tr.norms[idx] < norm_floor * 0.65 and weights.items.len >= 6) continue;
        const v = tr.unit[idx * dim ..][0..dim];
        var dup = false;
        for (0..weights.items.len) |j| {
            if (quant.dotF32(vecs.items[j * dim ..][0..dim], v) > dedup_cos) {
                dup = true;
                break;
            }
        }
        if (dup) continue;
        try vecs.appendSlice(gpa, v);
        try weights.append(gpa, tr.spec[idx]);
    }
    if (weights.items.len == 0) {
        const best = order[0];
        try vecs.appendSlice(gpa, tr.unit[best * dim ..][0..dim]);
        try weights.append(gpa, @max(tr.spec[best], 0.25));
    }
    const count = weights.items.len;
    return .{
        .vecs = try vecs.toOwnedSlice(gpa),
        .weights = try weights.toOwnedSlice(gpa),
        .count = count,
        .dim = dim,
    };
}

test "whitener: removes dominant background direction and assigns high specificity to novel vectors" {
    const gpa = std.testing.allocator;
    const dim = 32;
    const n_cal = 40;
    const raw = try gpa.alloc(f32, n_cal * dim);
    defer gpa.free(raw);
    for (0..n_cal) |i| {
        const row = raw[i * dim ..][0..dim];
        @memset(row, 0);
        // Strong shared background on dims 0..3
        row[0] = 10.0 + @as(f32, @floatFromInt(i % 5));
        row[1] = -8.0 + 0.3 * @as(f32, @floatFromInt(i % 7));
        row[2 + (i % 4)] = 2.0;
    }
    var w = try LayerWhitener.fit(gpa, raw, n_cal, dim, 4, 64);
    defer w.deinit(gpa);

    // Query with an orthogonal novel direction on dim 20 -> specificity should be ~1.0!
    var probe: [2 * dim]f32 = @splat(0);
    probe[0] = 11.0;
    probe[1] = -7.5;
    probe[2] = 2.0; // background-like
    probe[dim + 0] = 10.5;
    probe[dim + 1] = -7.5;
    probe[dim + 2] = 2.0;
    probe[dim + 20] = 5.0; // novel entity!
    var tr = try w.transform(gpa, &probe, 2);
    defer tr.deinit(gpa);
    try std.testing.expect(tr.spec[0] < 0.1);
    try std.testing.expect(tr.spec[1] > 0.85);
    try std.testing.expect(tr.spec[1] > tr.spec[0]);
}
