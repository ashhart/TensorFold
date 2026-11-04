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
    prompts: usize = 0,
    hidden: usize,

    pub fn init(a: Allocator, layers: usize, hidden: usize) !Mean {
        if (layers == 0 or hidden == 0) return error.BadMeanShape;
        return .{ .total = try a.alloc(f32, layers * hidden), .hidden = hidden };
    }

    pub fn deinit(m: *Mean, a: Allocator) void {
        a.free(m.total);
        m.total = &.{};
    }

    /// Adds one prompt's sums: its own mean, weighted like every other prompt's.
    pub fn addPrompt(m: *Mean, sums: []const f32, divisor: f32) !void {
        if (divisor <= 0 or sums.len != m.total.len) return error.BadPromptSums;
        for (m.total, sums) |*t, s| t.* += s / divisor;
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

/// The two sets' difference of means per layer, each unit-normalised; a layer with no difference is dropped.
pub fn derive(a: Allocator, harmful: Mean, harmless: Mean, layers: usize, hidden: usize) !Derived {
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
    for (0..layers) |id| {
        harmful.mean(id, left);
        harmless.mean(id, right);
        var sum: f64 = 0;
        for (left, right, diff) |l, r, *d| {
            d.* = l - r;
            sum += @as(f64, d.*) * @as(f64, d.*);
        }
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
