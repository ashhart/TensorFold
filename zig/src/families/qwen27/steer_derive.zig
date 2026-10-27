//! Control vectors from the model itself: the per-layer residual sums and the difference of means.
const std = @import("std");
const mtl = @import("metal");
const source = @import("steer_sources");

const Allocator = std.mem.Allocator;

/// The GLP metadata this producer writes: how the numbers were made and what they contrast.
pub const method = "difference_of_means_per_layer";
pub const contrast = "form_matched_harmful_vs_harmless_unpaired";

/// The method name for a capture that took the prompt's last row instead of all of them.
pub const method_last = "difference_of_means_last_token_per_layer";

/// Which rows of a prompt the capture takes: the prompt's whole content, or its last row.
pub const Site = enum {
    /// Every row's mean: the prompt's content, read out where the next word is decided.
    mean_rows,
    /// The last row alone: the one the next word is read from, with nothing of the content in it.
    last_row,

    /// The method name this site writes into the file.
    pub fn methodName(s: Site) []const u8 {
        return switch (s) {
            .mean_rows => method,
            .last_row => method_last,
        };
    }
};

/// The kernel's constants: the layer whose rows are summed, and the residual's width.
const Sum = extern struct { layer: u32, width: u32 };

/// The per-layer residual sums: one fp32 vector per layer, every captured prompt added in.
pub const Capture = struct {
    buffer: mtl.Buffer,
    dims: mtl.Buffer,
    library: mtl.Library,
    pipeline: mtl.Pipeline,
    layers: usize,
    hidden: usize,
    site: Site,

    /// The accumulator for `layers` layers of `hidden` columns; `dims` is the frame's row metadata.
    pub fn init(device: mtl.Device, layers: usize, hidden: usize, dims: mtl.Buffer, site: Site) !Capture {
        if (layers == 0 or hidden == 0) return error.BadCaptureShape;
        const library = try mtl.Library.fromSource(device, source.capture, mtl.CompileOptions.mlx());
        errdefer library.deinit();
        const kernel = switch (site) {
            .mean_rows => "q27_sumrows",
            .last_row => "q27_lastrow",
        };
        const pipeline = try mtl.Pipeline.init(device, library, kernel, false);
        errdefer pipeline.deinit();
        const buffer = try device.buffer(layers * hidden * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer buffer.deinit();
        return .{ .buffer = buffer, .dims = dims, .library = library, .pipeline = pipeline, .layers = layers, .hidden = hidden, .site = site };
    }

    pub fn deinit(c: *Capture) void {
        c.buffer.deinit();
        c.pipeline.deinit();
        c.library.deinit();
    }

    /// Zeroes the accumulator, so one prompt's sums stand alone.
    pub fn reset(c: *Capture) void {
        @memset(c.buffer.contents()[0..c.buffer.length()], 0);
    }

    /// Adds one layer's residual rows to that layer's accumulator; only the rows below dims[0] are real.
    pub fn record(c: *Capture, e: mtl.ComputeEncoder, layer: usize, residual: anytype) !void {
        if (layer >= c.layers) return error.BadCaptureLayer;
        e.setPipeline(c.pipeline);
        e.setBuffer(residual.buffer, residual.offset, 0);
        e.setBuffer(c.dims, 0, 1);
        e.setBuffer(c.buffer, 0, 2);
        e.setValue(Sum{ .layer = @intCast(layer), .width = @intCast(c.hidden) }, 3);
        e.dispatchThreads(.{ .width = c.hidden }, .{ .width = 256 });
        // The next layer writes the same residual buffer, so the reads here end first.
        e.barrier();
    }

    /// The frame's real row count: the rows the kernel summed.
    pub fn rows(c: *Capture) usize {
        return @intCast(c.dims.slice(i32, 8)[0]);
    }

    /// What one prompt's sums divide by: its row count for a mean of rows, and one for the last row.
    pub fn divisor(c: *Capture) f32 {
        return switch (c.site) {
            .mean_rows => @floatFromInt(c.rows()),
            .last_row => 1,
        };
    }

    /// The accumulator's host view: `layers` vectors of `hidden` sums.
    pub fn sums(c: *Capture) []const f32 {
        return c.buffer.slice(f32, c.layers * c.hidden);
    }
};

/// One prompt set's mean: every prompt's own mean, counted once.
pub const Mean = struct {
    /// The per-prompt means, summed: a long prompt does not outweigh a short one.
    total: []f32,
    /// The largest a dimension's value ever was over the set's prompts: the massive activations.
    peak: []f32,
    prompts: usize = 0,
    hidden: usize,

    pub fn init(a: Allocator, layers: usize, hidden: usize) !Mean {
        if (layers == 0 or hidden == 0) return error.BadMeanShape;
        const total = try a.alloc(f32, layers * hidden);
        errdefer a.free(total);
        const peak = try a.alloc(f32, layers * hidden);
        errdefer a.free(peak);
        @memset(peak, 0);
        return .{ .total = total, .peak = peak, .hidden = hidden };
    }

    pub fn deinit(m: *Mean, a: Allocator) void {
        a.free(m.total);
        a.free(m.peak);
        m.total = &.{};
        m.peak = &.{};
    }

    /// Adds one prompt's sums: its own mean, weighted like every other prompt's.
    pub fn addPrompt(m: *Mean, sums: []const f32, divisor: f32) !void {
        if (divisor <= 0 or sums.len != m.total.len) return error.BadPromptSums;
        for (m.total, m.peak, sums) |*t, *p, s| {
            const v = s / divisor;
            t.* += v;
            p.* = @max(p.*, @abs(v));
        }
        m.prompts += 1;
    }

    /// The set's mean for one layer, written into `out`.
    pub fn mean(m: Mean, id: usize, out: []f32) void {
        const count: f32 = @floatFromInt(@max(m.prompts, 1));
        const base = id * m.hidden;
        for (out, m.total[base..][0..m.hidden]) |*o, v| o.* = v / count;
    }
};

/// One layer's direction: the difference of the two sets' means, unit-normalised.
pub const Layer = struct {
    id: u32,
    dir: []f32,
};

/// The derived directions: one unit vector per layer, and the layers whose difference was degenerate.
pub const Derived = struct {
    layers: []Layer,
    dropped: []u32,

    pub fn deinit(d: *Derived, a: Allocator) void {
        for (d.layers) |l| a.free(l.dir);
        a.free(d.layers);
        a.free(d.dropped);
        d.* = .{ .layers = &.{}, .dropped = &.{} };
    }
};

/// The two sets' peaks in one layer: the largest a dimension's activation ever was, over both.
const Peaks = struct {
    harmful: Mean,
    harmless: Mean,
    base: usize,

    fn at(p: Peaks, i: usize) f32 {
        return @max(p.harmful.peak[p.base + i], p.harmless.peak[p.base + i]);
    }
};

/// The dims a layer's direction leaves out: the top `frac` of them by the activation they carry.
fn maskDims(peaks: Peaks, hidden: usize, frac: f32, order: []usize, masked: []bool) void {
    @memset(masked, false);
    const count = @as(usize, @intFromFloat(@round(frac * @as(f32, @floatFromInt(hidden)))));
    const k = @min(hidden, @max(1, count));
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order, peaks, struct {
        fn lessThan(p: Peaks, x: usize, y: usize) bool {
            const px = p.at(x);
            const py = p.at(y);
            if (px != py) return px > py;
            return x < y;
        }
    }.lessThan);
    for (order[0..k]) |i| masked[i] = true;
}

/// The peak-to-median ratio of a layer's activations: how far the massive activations stand out.
pub fn massiveRatio(harmful: Mean, harmless: Mean, id: usize, hidden: usize) f64 {
    const peaks = Peaks{ .harmful = harmful, .harmless = harmless, .base = id * hidden };
    var values: [5120]f32 = undefined;
    if (hidden > values.len) return 0;
    for (0..hidden) |i| values[i] = peaks.at(i);
    std.mem.sort(f32, values[0..hidden], {}, std.sort.asc(f32));
    const median = values[hidden / 2];
    if (median <= 0) return 0;
    return @as(f64, values[hidden - 1]) / @as(f64, median);
}

/// The two sets' difference of means per layer, each unit-normalised; a layer with no difference is dropped.
pub fn derive(a: Allocator, harmful: Mean, harmless: Mean, layers: usize, hidden: usize, mask: f32) !Derived {
    var kept: std.ArrayList(Layer) = .empty;
    errdefer {
        for (kept.items) |l| a.free(l.dir);
        kept.deinit(a);
    }
    var dropped: std.ArrayList(u32) = .empty;
    errdefer dropped.deinit(a);
    const left = try a.alloc(f32, hidden);
    defer a.free(left);
    const right = try a.alloc(f32, hidden);
    defer a.free(right);
    const diff = try a.alloc(f32, hidden);
    defer a.free(diff);
    const order = try a.alloc(usize, hidden);
    defer a.free(order);
    const masked = try a.alloc(bool, hidden);
    defer a.free(masked);
    for (0..layers) |id| {
        harmful.mean(id, left);
        harmless.mean(id, right);
        for (left, right, diff) |l, r, *d| d.* = l - r;
        if (mask > 0) {
            const peaks = Peaks{ .harmful = harmful, .harmless = harmless, .base = id * hidden };
            maskDims(peaks, hidden, mask, order, masked);
            for (diff, masked) |*d, m| {
                if (m) d.* = 0;
            }
        }
        var sum: f64 = 0;
        for (diff) |d| sum += @as(f64, d) * @as(f64, d);
        const norm = @sqrt(sum);
        if (!std.math.isFinite(norm) or norm == 0) {
            try dropped.append(a, @intCast(id));
            continue;
        }
        var finite = true;
        for (diff) |*d| {
            d.* = @floatCast(@as(f64, d.*) / norm);
            if (!std.math.isFinite(d.*)) finite = false;
        }
        if (!finite) {
            try dropped.append(a, @intCast(id));
            continue;
        }
        const dir = try a.alloc(f32, hidden);
        errdefer a.free(dir);
        @memcpy(dir, diff);
        try kept.append(a, .{ .id = @intCast(id), .dir = dir });
    }
    const layers_out = try kept.toOwnedSlice(a);
    errdefer {
        for (layers_out) |l| a.free(l.dir);
        a.free(layers_out);
    }
    return .{ .layers = layers_out, .dropped = try dropped.toOwnedSlice(a) };
}
