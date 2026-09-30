const std = @import("std");
const mx = @import("mlx.zig");
const policy = @import("memory_budget.zig");
const session = @import("session.zig");

pub fn activeBytes() !u64 {
    var value: usize = 0;
    try mx.check(mx.c.mlx_get_active_memory(&value));
    return value;
}

pub fn recommendedBytes() !usize {
    const device = mx.c.mlx_device_new_type(mx.c.MLX_GPU, 0);
    defer _ = mx.c.mlx_device_free(device);
    var info = mx.c.mlx_device_info_new();
    defer _ = mx.c.mlx_device_info_free(info);
    try mx.check(mx.c.mlx_device_info_get(&info, device));
    var value: usize = 0;
    try mx.check(mx.c.mlx_device_info_get_size(&value, info, "max_recommended_working_set_size"));
    return value;
}

pub const Runtime = struct {
    ram: u64,
    budget: u64,
    fraction: f64,
    share: u64,
    previous_memory: usize,
    previous_cache: usize,
    previous_wired: ?usize = null,

    pub fn init(override: ?[]const u8, glm: bool) !Runtime {
        var ram: u64 = 0;
        var size: usize = @sizeOf(u64);
        if (std.c.sysctlbyname("hw.memsize", &ram, &size, null, 0) != 0 or ram == 0) return error.PhysicalMemoryUnavailable;
        const fraction: f64 = if (glm and ram <= 256 * policy.gib) 0.85 else 0.70;
        const budget = try policy.limit(ram, try recommendedBytes(), fraction, override);
        const share = @max(1, budget -| policy.process_bytes);
        var previous_memory: usize = 0;
        try mx.check(mx.c.mlx_set_memory_limit(&previous_memory, @intCast(share)));
        errdefer _ = mx.c.mlx_set_memory_limit(&previous_memory, previous_memory);
        var previous_cache: usize = 0;
        try mx.check(mx.c.mlx_set_cache_limit(&previous_cache, @intCast(@min(2 * policy.gib, share))));
        return .{ .ram = ram, .budget = budget, .fraction = fraction, .share = share, .previous_memory = previous_memory, .previous_cache = previous_cache };
    }

    pub fn deinit(runtime: *Runtime) void {
        _ = mx.c.mlx_synchronize(mx.stream);
        var ignored: usize = 0;
        if (runtime.previous_wired) |previous| _ = mx.c.mlx_set_wired_limit(&ignored, previous);
        _ = mx.c.mlx_set_cache_limit(&ignored, runtime.previous_cache);
        _ = mx.c.mlx_set_memory_limit(&ignored, runtime.previous_memory);
    }

    pub fn checkWeights(runtime: Runtime, io: std.Io, directory: []const u8, flash: bool) !void {
        return runtime.checkWeightsAndDraft(io, directory, flash, null);
    }

    pub fn checkWeightsAndDraft(runtime: Runtime, io: std.Io, directory: []const u8, flash: bool, drafter: ?[]const u8) !void {
        var size = try weightBytes(io, directory, flash);
        if (drafter) |path| size = try std.math.add(u64, size, try weightBytes(io, path, false));
        if (size >= runtime.share) {
            std.debug.print("Weights need {d:.1} GiB; the server budget leaves {d:.1} GiB for MLX. Set TENSORFOLD_MEMORY_LIMIT_GB within this Mac's working-set limit or use a smaller checkpoint.\n", .{ @as(f64, @floatFromInt(size)) / policy.gib, @as(f64, @floatFromInt(runtime.share)) / policy.gib });
            return error.WeightsExceedMemoryBudget;
        }
    }

    fn weightBytes(io: std.Io, directory: []const u8, flash: bool) !u64 {
        var dir = try std.Io.Dir.cwd().openDir(io, directory, .{ .iterate = true });
        defer dir.close(io);
        var iterator = dir.iterate();
        var size: u64 = 0;
        while (try iterator.next(io)) |entry| {
            if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
            size = try std.math.add(u64, size, (try dir.statFile(io, entry.name, .{})).size);
            if (flash) {
                const path = try std.fs.path.join(mx.allocator, &.{ directory, entry.name });
                defer mx.allocator.free(path);
                var file = try @import("safetensors.zig").File.open(mx.allocator, io, path);
                defer file.deinit();
                var tensors = file.header.tensors.iterator();
                while (tensors.next()) |tensor| if (std.mem.startsWith(u8, tensor.key_ptr.*, "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_")) {
                    size -= tensor.value_ptr.len;
                };
            }
        }
        return size;
    }

    pub fn wire(runtime: *Runtime) !void {
        try mx.check(mx.c.mlx_synchronize(mx.stream));
        try mx.check(mx.c.mlx_clear_cache());
        var ceiling = try recommendedBytes();
        var mb: u64 = 0;
        var size: usize = @sizeOf(u64);
        if (std.c.sysctlbyname("iogpu.wired_limit_mb", &mb, &size, null, 0) == 0 and mb > 0) ceiling = @min(ceiling, mb *| (1024 * 1024));
        const limit = @min(try activeBytes(), @min(runtime.budget, ceiling));
        if (limit == 0) return;
        var previous: usize = 0;
        if (mx.c.mlx_set_wired_limit(&previous, @intCast(limit)) == 0) runtime.previous_wired = previous;
    }

    pub fn admissionBudget(runtime: Runtime, io: std.Io) !u64 {
        const a = mx.allocator;
        const result = std.process.run(a, io, .{ .argv = &.{"/usr/bin/memory_pressure"}, .stdout_limit = .limited(64 * 1024), .stderr_limit = .limited(4096) }) catch return runtime.share;
        defer a.free(result.stdout);
        defer a.free(result.stderr);
        const marker = "free percentage:";
        const at = std.mem.indexOf(u8, result.stdout, marker) orelse return runtime.share;
        const rest = std.mem.trimStart(u8, result.stdout[at + marker.len ..], " \t");
        var end: usize = 0;
        while (end < rest.len and std.ascii.isDigit(rest[end])) : (end += 1) {}
        const free = std.fmt.parseInt(u64, rest[0..end], 10) catch return runtime.share;
        if (free > 100) return runtime.share;
        var cached: usize = 0;
        try mx.check(mx.c.mlx_get_cache_memory(&cached));
        const elsewhere = @as(u64, @intFromFloat(@as(f64, @floatFromInt(runtime.ram * (100 - free))) / 100)) -| (try activeBytes() + cached);
        return policy.concurrentBudget(runtime.ram, runtime.fraction, runtime.budget, runtime.share, elsewhere);
    }
};

fn arrayBytes(array: mx.Array) u64 {
    return if (array.ctx == null) 0 else mx.c.mlx_array_nbytes(array);
}

fn perPosition(array: mx.Array, axis: usize) f64 {
    if (array.ctx == null) return 0;
    const positions = mx.shape(array)[axis];
    return if (positions <= 0) 0 else @as(f64, @floatFromInt(arrayBytes(array))) / @as(f64, @floatFromInt(positions));
}

fn growthFloor(comptime M: type, state: *@import("request_state.zig").State(M)) struct { kv: f64, spare: f64 } {
    var kv: f64 = 0;
    var spare: f64 = 0;
    for (state.cache, 0..) |cache, index| {
        if (@TypeOf(cache.keys) == @import("kv_buffer.zig").Buffer) {
            inline for (.{ "keys", "values", "index_keys" }) |name| if (@hasField(@TypeOf(cache), name)) {
                const buffer = @field(cache, name);
                const each = perPosition(buffer.current, buffer.axis);
                kv += each;
                spare += each;
            };
            if (@hasField(@TypeOf(cache), "pooled")) {
                kv += @as(f64, @floatFromInt(arrayBytes(cache.pooled) + arrayBytes(cache.token_history))) / @as(f64, @floatFromInt(@max(1, state.position)));
            }
        } else if (M == @import("gemma.zig").Model) {
            if (index % 6 == 5) kv += perPosition(cache.keys, 2) + perPosition(cache.values, 2);
        } else if (M == @import("glm.zig").Model) {
            inline for (.{ "keys", "ik", "ig", "pool" }) |name| kv += @as(f64, @floatFromInt(arrayBytes(@field(cache, name)))) / @as(f64, @floatFromInt(@max(1, state.position)));
        } else if (M == @import("deepseek.zig").Model) {
            kv += @as(f64, @floatFromInt(arrayBytes(cache.pool) + arrayBytes(cache.ipool))) / @as(f64, @floatFromInt(@max(1, state.position)));
        }
    }
    if (@hasDecl(M, "DraftCache")) {
        inline for (comptime std.meta.fieldNames(M.DraftCache)) |name| {
            if (@FieldType(M.DraftCache, name) == @import("kv_buffer.zig").Buffer) {
                const buffer = @field(state.head_cache, name);
                const each = perPosition(buffer.current, buffer.axis);
                kv += each;
                spare += each;
            }
        }
    }
    return .{ .kv = kv, .spare = spare };
}

pub fn measure(s: *session.Session) !policy.StreamMemory {
    switch (s.backend) {
        inline else => |*m| return measureModel(m, &s.tokenizer, s.draftSink(.{})),
    }
}

fn measureModel(m: anytype, tokenizer: *@import("vendor/tokenizer.zig").Tokenizer, sink: session.Sink) !policy.StreamMemory {
    const M = @TypeOf(m.*);
    const G = session.Generation(M);
    const chunk: usize = if (@hasDecl(M, "prefill")) 2048 else 16;
    const vocab: usize = if (M == @import("model.zig").Model) 248320 else if (@hasField(M, "vocab")) @intCast(m.vocab) else M.vocab;
    var tokens: [2 * chunk + 64]i32 = undefined;
    for (&tokens, 0..) |*token, index| token.* = @intCast((1000 + index) % vocab);
    var held: [3]G = undefined;
    var initialized: usize = 0;
    defer {
        for (held[0..initialized]) |*generation| generation.deinit();
        _ = mx.c.mlx_clear_cache();
    }
    // Materialize the same path before measuring cache growth.
    {
        var warm = try G.init(m, tokenizer, mx.allocator, tokens[0..64], .{ .max_tokens = 1 }, sink, null);
        defer warm.deinit();
        while (warm.phase == .prefill) _ = try warm.step(m);
    }
    var sizes: [3]u64 = undefined;
    var peaks: [3]u64 = undefined;
    const probes = [_]usize{ 64, chunk + 64, 2 * chunk + 64 };
    for (probes, &held, &sizes, &peaks) |count, *generation, *size, *peak| {
        try mx.check(mx.c.mlx_clear_cache());
        const before = try activeBytes();
        try mx.check(mx.c.mlx_reset_peak_memory());
        generation.* = try G.init(m, tokenizer, mx.allocator, tokens[0..count], .{ .max_tokens = @max(2, sink.draft_budget + 2), .ignore_eos = true }, sink, null);
        initialized += 1;
        while (generation.phase == .prefill) _ = try generation.step(m);
        try mx.check(mx.c.mlx_synchronize(mx.stream));
        const after = try activeBytes();
        var high: usize = 0;
        try mx.check(mx.c.mlx_get_peak_memory(&high));
        size.* = after -| before;
        peak.* = high -| after;
    }
    const floor = growthFloor(M, &held[2].state);
    const per_token = @max((@as(f64, @floatFromInt(sizes[2])) - @as(f64, @floatFromInt(sizes[1]))) / chunk, floor.kv + floor.spare);
    const spare: u64 = @intFromFloat(floor.spare * 2048);
    const b = @max(0, (@as(f64, @floatFromInt(peaks[2])) - @as(f64, @floatFromInt(peaks[1]))) / (chunk * chunk));
    const a = @max(0, @as(f64, @floatFromInt(peaks[1])) / chunk - b * chunk);
    const before_decode = try activeBytes();
    try mx.check(mx.c.mlx_reset_peak_memory());
    _ = try held[2].step(m);
    try mx.check(mx.c.mlx_synchronize(mx.stream));
    var decode_peak: usize = 0;
    try mx.check(mx.c.mlx_get_peak_memory(&decode_peak));
    return .{ .short_tokens = probes[0], .short = sizes[0] + spare, .long_tokens = probes[1], .long = @max(sizes[0], sizes[1]) + spare, .per_token = per_token, .prefill_a = a, .prefill_b = b, .round_bytes = decode_peak -| before_decode, .chunk = chunk };
}

pub fn check(io: std.Io, directory: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var runtime = try Runtime.init(null, false);
    defer runtime.deinit();
    var model = try session.Session.init(io, directory);
    defer model.deinit();
    const profile = try measure(&model);
    try profile.validate();
    try runtime.wire();
    std.debug.print("Memory profile: {d} bytes/token, short={d}, long={d}, chunk={d}, decode={d}, prefill a={d}, b={d}\n", .{ profile.per_token, profile.short, profile.long, profile.chunk, profile.round_bytes, profile.prefill_a, profile.prefill_b });
    switch (model.backend) {
        inline else => |*m| {
            const M = @TypeOf(m.*);
            var tokens: [4161]i32 = undefined;
            for (&tokens, 0..) |*token, index| token.* = @intCast(1000 + index);
            const counts: []const usize = if (@hasDecl(M, "prefill")) &.{ 1, 65, 2047, 2048, 2049, 4161 } else &.{ 1, 15, 16, 17, 65, 257 };
            for (counts) |count| {
                try mx.check(mx.c.mlx_clear_cache());
                const before = try activeBytes();
                var generation = try session.Generation(M).init(m, &model.tokenizer, mx.allocator, tokens[0..count], .{ .max_tokens = 2, .ignore_eos = true }, .{}, null);
                defer generation.deinit();
                while (!try generation.step(m)) {}
                try mx.check(mx.c.mlx_synchronize(mx.stream));
                const held = (try activeBytes()) -| before;
                const estimate = try profile.streamBytes(count + 2);
                if (held > estimate) {
                    std.debug.print("Underestimated {d}-token request: {d} held, {d} predicted\n", .{ count, held, estimate });
                    return error.CacheMemoryUnderestimated;
                }
                if (m.position != 0) return error.ProbeChangedModelState;
            }
        },
    }
    std.debug.print("PASS: request memory probes cover prefill boundaries and decode growth without retaining request state\n", .{});
}
