//! A captured set of Triton cubins (aot.json + cubins/): each launch picks the variant Triton itself would have picked.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
const launch = @import("launch.zig");
const triton = @import("triton.zig");

/// Where a family's glue kernels come from: our own fatbins, or a captured set in a folder.
pub const Pick = union(enum) { native, captured: []const u8, missing: []const u8 };

/// `explicit` (folder or native), else the binary's share/tensorfold/cuda/sm<capability> when it has aot.json.
pub fn pick(a: std.mem.Allocator, io: std.Io, explicit: ?[]const u8, capability: u32) !Pick {
    if (explicit) |dir| {
        if (std.mem.eql(u8, dir, "native")) return .native;
        return if (has(a, io, dir)) .{ .captured = dir } else .{ .missing = dir };
    }
    const exe = try std.process.executableDirPathAlloc(io, a);
    const dir = try std.fs.path.join(a, &.{ exe, "..", "share", "tensorfold", "cuda", try std.fmt.allocPrint(a, "sm{d}", .{capability}) });
    return if (has(a, io, dir)) .{ .captured = dir } else .native;
}

fn has(a: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    const path = std.fs.path.join(a, &.{ dir, "aot.json" }) catch return false;
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

const ParamJson = struct { name: []const u8, type: []const u8, div16: bool, nospec: bool };
const ConstJson = struct { int: ?i64 = null, f32: ?u32 = null };
const KernelJson = struct {
    @"fn": []const u8,
    hash: []const u8,
    name: []const u8,
    num_warps: u32,
    num_ctas: u32 = 1,
    shared: u32 = 0,
    global_scratch: u32 = 0,
    global_align: u32 = 1,
    profile_scratch: u32 = 0,
    pdl: bool = false,
    params: []ParamJson,
    consts: std.json.ArrayHashMap(ConstJson),
};
const SetJson = struct { kernels: []KernelJson };

/// A variant's description in aot.json, and a whole set's: what `Set.find` matches launches against.
pub const Spec = KernelJson;
pub const SpecSet = SetJson;

/// aot.json parsed without loading any cubin (host-side checks of variant selection).
pub fn parseSpecs(gpa: std.mem.Allocator, text: []const u8) !std.json.Parsed(SpecSet) {
    return std.json.parseFromSlice(SpecSet, gpa, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
}

/// Whether variant `k` is the one Triton would launch for these arguments and constexprs (`Set.find`'s test).
pub fn specMatches(k: Spec, args: []const Arg, consts: []const Const) bool {
    return matches(k, args, consts);
}

/// One argument of a launch, by the kernel's parameter name.
pub const Arg = struct {
    name: []const u8,
    value: Value,

    pub const Value = union(enum) { ptr: struct { addr: u64, ty: []const u8 }, i32: i32, f32: f32, u64: u64 };
};

/// A constexpr the call site compiled the kernel with (Python's keyword arguments): ints, bools, fp32 bits.
pub const Const = struct { name: []const u8, int: ?i64 = null, f32: ?f32 = null };

pub fn ptr(name: []const u8, ty: []const u8, addr: u64) Arg {
    return .{ .name = name, .value = .{ .ptr = .{ .addr = addr, .ty = ty } } };
}
pub fn int(name: []const u8, v: i32) Arg {
    return .{ .name = name, .value = .{ .i32 = v } };
}
pub fn float(name: []const u8, v: f32) Arg {
    return .{ .name = name, .value = .{ .f32 = v } };
}
pub fn word(name: []const u8, v: u64) Arg {
    return .{ .name = name, .value = .{ .u64 = v } };
}
pub fn ci(name: []const u8, v: i64) Const {
    return .{ .name = name, .int = v };
}
pub fn cf(name: []const u8, v: f32) Const {
    return .{ .name = name, .f32 = v };
}

const Variant = struct { spec: KernelJson, kernel: triton.Kernel };

pub const Set = struct {
    parsed: std.json.Parsed(SetJson),
    variants: []Variant,
    gpa: std.mem.Allocator,
    /// every distinct lookup key and its outcome, when the caller attaches a Trace (the set destroys it)
    trace: ?*Trace = null,

    /// Loads every cubin listed in `dir`/aot.json into its own module.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, device: abi.Device, dir: []const u8) !Set {
        const path = try std.fs.path.join(gpa, &.{ dir, "aot.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
        defer gpa.free(text);
        const parsed = try parseSpecs(gpa, text);
        errdefer parsed.deinit();
        const variants = try gpa.alloc(Variant, parsed.value.kernels.len);
        var n: usize = 0;
        errdefer {
            for (variants[0..n]) |*v| v.kernel.unload();
            gpa.free(variants);
        }
        for (parsed.value.kernels) |k| {
            if (k.global_scratch != 0 or k.profile_scratch != 0) return error.ScratchUnsupported;
            const file = try std.fmt.allocPrint(gpa, "{s}/cubins/{s}.cubin", .{ dir, k.hash });
            defer gpa.free(file);
            const cubin = try std.Io.Dir.cwd().readFileAllocOptions(io, file, gpa, .limited(1 << 26), .@"16", null);
            defer gpa.free(cubin);
            const name_z = try gpa.dupeSentinel(u8, k.name, 0);
            defer gpa.free(name_z);
            const meta: triton.Meta = .{ .name = k.name, .num_warps = k.num_warps, .num_ctas = k.num_ctas, .shared = k.shared, .launch_pdl = k.pdl };
            variants[n] = .{ .spec = k, .kernel = try triton.Kernel.load(d, device, cubin, meta, name_z) };
            n += 1;
        }
        return .{ .parsed = parsed, .variants = variants, .gpa = gpa };
    }

    pub fn deinit(self: *Set) void {
        if (self.trace) |t| t.destroy();
        for (self.variants) |*v| v.kernel.unload();
        self.gpa.free(self.variants);
        self.parsed.deinit();
        self.* = undefined;
    }

    /// The variant of `function` compiled for these constexprs and these arguments' specialization.
    pub fn find(self: *const Set, function: []const u8, args: []const Arg, consts: []const Const) !*const Variant {
        for (self.variants) |*v| {
            if (!std.mem.eql(u8, v.spec.@"fn", function)) continue;
            if (!matches(v.spec, args, consts)) continue;
            if (self.trace) |t| t.note(function, args, consts, v.spec.hash);
            return v;
        }
        if (self.trace) |t| t.note(function, args, consts, null);
        std.log.err("no captured Triton variant of {s} for this launch:", .{function});
        for (args) |a| switch (a.value) {
            .ptr => |p| std.log.err("  {s}: {s} at {x} (16-aligned {})", .{ a.name, p.ty, p.addr, p.addr % 16 == 0 }),
            .i32 => |x| std.log.err("  {s}: i32 {d}", .{ a.name, x }),
            .f32 => |x| std.log.err("  {s}: fp32 {d}", .{ a.name, x }),
            .u64 => |x| std.log.err("  {s}: u64 {d}", .{ a.name, x }),
        };
        // the constexprs pick the variant as much as the arguments do: without them the gap cannot be named
        for (consts) |c| {
            if (c.int) |x| std.log.err("  constexpr {s} = {d}", .{ c.name, x }) else if (c.f32) |x| std.log.err("  constexpr {s} = {d}", .{ c.name, x });
        }
        return error.MissingTritonVariant;
    }

    /// The smallest value of constexpr `name` at or above `at_least` among `function`'s variants.
    pub fn smallestConst(self: *const Set, function: []const u8, name: []const u8, at_least: i64) ?i64 {
        var best: ?i64 = null;
        for (self.variants) |v| {
            if (!std.mem.eql(u8, v.spec.@"fn", function)) continue;
            const c = (v.spec.consts.map.get(name) orelse continue).int orelse continue;
            if (c >= at_least and (best == null or c < best.?)) best = c;
        }
        return best;
    }

    /// Launches `function` on `grid` as Triton's launcher would: runtime arguments in the variant's order, scratch null.
    pub fn run(self: *const Set, stream: Stream, function: []const u8, grid: [3]u32, args: []const Arg, consts: []const Const) !void {
        const v = try self.find(function, args, consts);
        var packed_args: launch.Args = .{};
        for (v.spec.params) |p| {
            const a = lookup(args, p.name).?;
            switch (a.value) {
                .ptr => |x| packed_args.add(x.addr),
                .i32 => |x| packed_args.add(x),
                .f32 => |x| packed_args.add(x),
                .u64 => |x| packed_args.add(x),
            }
        }
        try v.kernel.launchOn(.{ .x = grid[0], .y = grid[1], .z = grid[2] }, stream, &packed_args, .{}, &.{});
    }
};

/// One JSON line per distinct `find` key (as `matches` reads it) and its hit hash or null, rewritten to `path`.
pub const Trace = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    mutex: std.Io.Mutex = .init,
    seen: std.AutoHashMapUnmanaged(u64, void) = .empty,
    text: std.ArrayList(u8) = .empty,

    pub fn create(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !*Trace {
        const t = try gpa.create(Trace);
        errdefer gpa.destroy(t);
        t.* = .{ .gpa = gpa, .io = io, .path = try gpa.dupe(u8, path) };
        return t;
    }

    pub fn destroy(t: *Trace) void {
        t.seen.deinit(t.gpa);
        t.text.deinit(t.gpa);
        t.gpa.free(t.path);
        t.gpa.destroy(t);
    }

    /// The key's hash: what `matches` tells apart, nothing else (equal keys pick the same variant).
    pub fn key(function: []const u8, args: []const Arg, consts: []const Const) u64 {
        var h = std.hash.Wyhash.init(0);
        h.update(function);
        for (args) |a| {
            h.update(a.name);
            h.update(class(a));
            if (a.value == .ptr) h.update(a.value.ptr.ty);
        }
        for (consts) |c| {
            h.update(c.name);
            if (c.int) |x| h.update(std.mem.asBytes(&x));
            if (c.f32) |x| h.update(std.mem.asBytes(&@as(u32, @bitCast(x))));
            h.update("|");
        }
        return h.final();
    }

    fn class(a: Arg) []const u8 {
        return switch (a.value) {
            .i32 => |x| if (x == 1) "1" else if (@mod(x, 16) == 0) "div16" else "int",
            .ptr => |x| if (x.addr % 16 == 0) "ptr/16" else "ptr",
            .f32 => "fp32",
            .u64 => |x| if (x % 16 == 0) "u64/16" else "u64",
        };
    }

    pub fn note(t: *Trace, function: []const u8, args: []const Arg, consts: []const Const, hash: ?[]const u8) void {
        const k = key(function, args, consts);
        t.mutex.lockUncancelable(t.io);
        defer t.mutex.unlock(t.io);
        const got = t.seen.getOrPut(t.gpa, k) catch return;
        if (got.found_existing) return;
        t.line(function, args, consts, hash) catch return;
        std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = t.path, .data = t.text.items }) catch {};
    }

    fn line(t: *Trace, function: []const u8, args: []const Arg, consts: []const Const, hash: ?[]const u8) !void {
        var w: std.Io.Writer.Allocating = .fromArrayList(t.gpa, &t.text);
        defer t.text = w.toArrayList();
        const o = &w.writer;
        try o.print("{{\"fn\":\"{s}\",\"hit\":", .{function});
        if (hash) |x| try o.print("\"{s}\"", .{x}) else try o.writeAll("null");
        try o.writeAll(",\"args\":{");
        for (args, 0..) |a, i| {
            const ty = switch (a.value) {
                .ptr => |x| x.ty,
                else => "",
            };
            try o.print("{s}\"{s}\":\"{s}{s}{s}\"", .{ if (i > 0) "," else "", a.name, ty, if (ty.len > 0) " " else "", class(a) });
        }
        try o.writeAll("},\"consts\":{");
        for (consts, 0..) |c, i| {
            try o.print("{s}\"{s}\":", .{ if (i > 0) "," else "", c.name });
            if (c.int) |x| try o.print("{d}", .{x}) else if (c.f32) |x| try o.print("{e}", .{x}) else try o.writeAll("null");
        }
        try o.writeAll("}}\n");
    }
};

fn lookup(args: []const Arg, name: []const u8) ?Arg {
    for (args) |a| if (std.mem.eql(u8, a.name, name)) return a;
    return null;
}

fn matches(k: KernelJson, args: []const Arg, consts: []const Const) bool {
    for (consts) |c| {
        const got = k.consts.map.get(c.name) orelse return false;
        if (c.int) |x| if (got.int == null or got.int.? != x) return false;
        if (c.f32) |x| if (got.f32 == null or got.f32.? != @as(u32, @bitCast(x))) return false;
    }
    var runtime: usize = 0;
    for (args) |a| {
        const param = for (k.params) |p| {
            if (std.mem.eql(u8, p.name, a.name)) break p;
        } else null;
        switch (a.value) {
            .i32 => |x| {
                if (param == null) {
                    // an int Triton folded: only the value 1 is ever specialized to a constexpr
                    const got = k.consts.map.get(a.name) orelse return false;
                    if (x != 1 or got.int == null or got.int.? != 1) return false;
                    continue;
                }
                const p = param.?;
                if (!std.mem.eql(u8, p.type, "i32")) return false;
                if (!p.nospec and x == 1) return false;
                if (p.div16 != (!p.nospec and @mod(x, 16) == 0)) return false;
            },
            .ptr => |x| {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, x.ty) or p.div16 != (x.addr % 16 == 0)) return false;
            },
            .f32 => {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, "fp32")) return false;
            },
            .u64 => |x| {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, "u64") or p.div16 != (x % 16 == 0)) return false;
            },
        }
        runtime += 1;
    }
    return runtime == k.params.len;
}

// -- tests: variant selection over an aot.json fixture, and the trace's one line per distinct key -------------------

const fixture =
    \\{"kernels": [
    \\  {"fn": "add", "hash": "a16", "name": "add_k", "num_warps": 4, "extra": "ignored",
    \\   "params": [{"name": "x", "type": "fp16", "div16": true, "nospec": false},
    \\              {"name": "n", "type": "i32", "div16": true, "nospec": false}],
    \\   "consts": {"BLOCK": {"int": 128}}},
    \\  {"fn": "add", "hash": "a1", "name": "add_k", "num_warps": 4,
    \\   "params": [{"name": "x", "type": "fp16", "div16": false, "nospec": false},
    \\              {"name": "n", "type": "i32", "div16": false, "nospec": false}],
    \\   "consts": {"BLOCK": {"int": 128}}},
    \\  {"fn": "add", "hash": "one", "name": "add_k", "num_warps": 4,
    \\   "params": [{"name": "x", "type": "fp16", "div16": true, "nospec": false}],
    \\   "consts": {"BLOCK": {"int": 128}, "n": {"int": 1}}}
    \\]}
;

fn picked(set: SpecSet, args: []const Arg, consts: []const Const) ?[]const u8 {
    for (set.kernels) |k| if (specMatches(k, args, consts)) return k.hash;
    return null;
}

test "aot specs: Triton's specialization picks the variant; a launch no variant covers picks none" {
    const t = std.testing;
    const p = try parseSpecs(t.allocator, fixture);
    defer p.deinit();
    try t.expectEqual(@as(usize, 3), p.value.kernels.len);
    try t.expectEqual(@as(u32, 1), p.value.kernels[0].num_ctas); // a field aot.json leaves out takes its default
    const block = [_]Const{ci("BLOCK", 128)};
    try t.expectEqualStrings("a16", picked(p.value, &.{ ptr("x", "fp16", 0x1000), int("n", 64) }, &block).?);
    try t.expectEqualStrings("a1", picked(p.value, &.{ ptr("x", "fp16", 0x1004), int("n", 63) }, &block).?);
    // n == 1 was folded into a constexpr when Triton compiled it: only the variant without an n parameter takes it
    try t.expectEqualStrings("one", picked(p.value, &.{ ptr("x", "fp16", 0x1000), int("n", 1) }, &block).?);
    // an unaligned pointer with a 16-divisible count, another dtype, another constexpr: no variant was captured
    try t.expectEqual(@as(?[]const u8, null), picked(p.value, &.{ ptr("x", "fp16", 0x1004), int("n", 64) }, &block));
    try t.expectEqual(@as(?[]const u8, null), picked(p.value, &.{ ptr("x", "bf16", 0x1000), int("n", 64) }, &block));
    try t.expectEqual(@as(?[]const u8, null), picked(p.value, &.{ ptr("x", "fp16", 0x1000), int("n", 64) }, &.{ci("BLOCK", 256)}));
}

test "aot trace: one line per distinct key, a launch that differs only in what matches ignores adds none" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    const path = try std.fs.path.join(t.allocator, &.{ dir, "trace.jsonl" });
    defer t.allocator.free(path);
    const tr = try Trace.create(t.allocator, t.io, path);
    defer tr.destroy();
    const block = [_]Const{ci("BLOCK", 128)};
    tr.note("add", &.{ ptr("x", "fp16", 0x1000), int("n", 64) }, &block, "a16");
    tr.note("add", &.{ ptr("x", "fp16", 0x2000), int("n", 32) }, &block, "a16"); // the same key: both 16-aligned
    tr.note("add", &.{ ptr("x", "fp16", 0x1004), int("n", 64) }, &block, null);
    const got = try std.Io.Dir.cwd().readFileAlloc(t.io, path, t.allocator, .limited(1 << 16));
    defer t.allocator.free(got);
    try t.expectEqualStrings(
        \\{"fn":"add","hit":"a16","args":{"x":"fp16 ptr/16","n":"div16"},"consts":{"BLOCK":128}}
        \\{"fn":"add","hit":null,"args":{"x":"fp16 ptr","n":"div16"},"consts":{"BLOCK":128}}
        \\
    , got);
}
