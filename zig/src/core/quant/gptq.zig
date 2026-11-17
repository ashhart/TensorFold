//! AutoGPTQ v1 weights (4-bit symmetric groups of 128, no act order): the rules, the stored tensors, an fp64 reference.
const std = @import("std");
const ckpt = @import("../checkpoint.zig");
const modelopt = @import("../modelopt.zig");
const Why = modelopt.Why;
const object = modelopt.object;
const text = modelopt.text;
const int = modelopt.int;

const Allocator = std.mem.Allocator;
const Checkpoint = ckpt.Checkpoint;
const Tensor = ckpt.Tensor;

/// The width and the group a config declares: anything else is refused by its field.
pub const bits = 4;
pub const group = 128;
/// The zero point of symmetric weights as qzeros stores it: 8 - 1, so a code reads as (nibble - 8).
pub const zero_nibble = 7;

/// What a `dynamic` rule does to the module its pattern matches.
pub const Kind = enum { quantize, skip };

/// One `dynamic` rule: the pattern after its `+:` or `-:`, which Python's re.match anchors at the start.
pub const Rule = struct { pattern: []const u8, kind: Kind };

/// A GPTQ quantization_config after admission, prepared to say which modules it stores quantized.
pub const Config = struct {
    arena: std.heap.ArenaAllocator,
    rules: []const Rule = &.{},

    /// The modules it quantizes: the last rule matching `module` decides, `default` where no rule does.
    pub fn quantizedDefault(self: *const Config, module: []const u8, default: bool) bool {
        var q = default;
        for (self.rules) |r| {
            if (regexMatch(r.pattern, module)) q = r.kind == .quantize;
        }
        return q;
    }

    /// The modules it quantizes, a module no rule names included.
    pub fn quantized(self: *const Config, module: []const u8) bool {
        return self.quantizedDefault(module, true);
    }

    pub fn deinit(self: *Config) void {
        self.arena.deinit();
    }
};

/// The GPTQ block of `config_json` admitted, else null when no `quantization_config` declares `quant_method` gptq.
pub fn admit(gpa: Allocator, config_json: []const u8, why: *Why) !?Config {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json, .{});
    if (parsed != .object) return null;
    return config(gpa, parsed.object, why);
}

/// `admit` over an already parsed config.json: the file's top-level object, whose text tower is read too.
pub fn config(gpa: Allocator, root: std.json.ObjectMap, why: *Why) !?Config {
    for ([_]?std.json.ObjectMap{ root, object(root.get("text_config")) }) |source| {
        const q = object((source orelse continue).get("quantization_config")) orelse continue;
        const method = text(q.get("quant_method")) orelse continue;
        if (!std.ascii.eqlIgnoreCase(method, "gptq")) continue;
        return try dialect(gpa, q, why);
    }
    return null;
}

/// The fields and the rules this reader reads; every one it does not is refused by its name.
fn dialect(gpa: Allocator, q: std.json.ObjectMap, why: *Why) !Config {
    const width = int(q.get("bits"));
    if (width != bits) {
        why.set("quantization_config.bits: {?d}, expected 4", .{width});
        return error.UnsupportedQuantization;
    }
    const size = int(q.get("group_size"));
    if (size != group) {
        why.set("quantization_config.group_size: {?d}, expected 128", .{size});
        return error.UnsupportedQuantization;
    }
    if (!holds(q.get("sym"), true)) {
        why.set("quantization_config.sym: {?}, expected true", .{flag(q.get("sym"))});
        return error.UnsupportedQuantization;
    }
    if (flag(q.get("desc_act"))) |act| if (act) {
        why.set("quantization_config.desc_act: true, this reader reads no act order", .{});
        return error.UnsupportedQuantization;
    };
    const dyn = object(q.get("dynamic")) orelse {
        const found = if (q.get("dynamic") == null) "absent" else "not an object";
        why.set("quantization_config.dynamic: {s}, this reader reads its rules", .{found});
        return error.UnsupportedQuantization;
    };
    var c: Config = .{ .arena = .init(gpa) };
    errdefer c.deinit();
    const a = c.arena.allocator();
    const rules = try a.alloc(Rule, dyn.count());
    var it = dyn.iterator();
    var i: usize = 0;
    while (it.next()) |kv| : (i += 1) {
        const key = kv.key_ptr.*;
        const kind: Kind = if (std.mem.startsWith(u8, key, "+:")) .quantize else if (std.mem.startsWith(u8, key, "-:")) .skip else {
            why.set("quantization_config.dynamic {s}: no +: or -: prefix", .{key});
            return error.UnsupportedQuantization;
        };
        if (!supported(key[2..])) {
            why.set("quantization_config.dynamic {s}: outside this reader's matcher", .{key});
            return error.UnsupportedQuantization;
        }
        try ruleMath(kv.value_ptr.*, key, why);
        rules[i] = .{ .pattern = try a.dupe(u8, key[2..]), .kind = kind };
    }
    c.rules = rules;
    if (!c.quantizedDefault("lm_head", holds(q.get("lm_head"), true))) {
        why.set("quantization_config.lm_head: {?}, the rules do not store the head int4", .{flag(q.get("lm_head"))});
        return error.UnsupportedQuantization;
    }
    return c;
}

/// A rule's own value: a width or a group other than the admitted one is mixed precision this reader does not read.
fn ruleMath(v: std.json.Value, key: []const u8, why: *Why) !void {
    if (v == .null) return;
    const o = object(v) orelse {
        why.set("quantization_config.dynamic {s}: not an object of bits", .{key});
        return error.UnsupportedQuantization;
    };
    if (int(o.get("bits"))) |b| if (b != bits) {
        why.set("quantization_config.dynamic {s}: bits {d}, expected 4", .{ key, b });
        return error.UnsupportedQuantization;
    };
    if (int(o.get("group_size"))) |g| if (g != group) {
        why.set("quantization_config.dynamic {s}: group_size {d}, expected 128", .{ key, g });
        return error.UnsupportedQuantization;
    };
}

/// Whether a pattern uses only what `regexMatch` reads, so a rule outside the subset is refused rather than misread.
fn supported(pattern: []const u8) bool {
    if (pattern.len == 0) return false;
    var i: usize = 0;
    var atom = false;
    while (i < pattern.len) : (i += 1) {
        switch (pattern[i]) {
            '\\' => {
                if (i + 1 >= pattern.len) return false;
                i += 1;
                atom = true;
            },
            '.' => atom = true,
            '*', '+' => {
                if (!atom) return false;
                atom = false;
            },
            '$' => return i + 1 == pattern.len,
            '?', '[', ']', '(', ')', '{', '}', '|', '^' => return false,
            else => atom = true,
        }
    }
    return true;
}

/// Python's re.match for the subset the rules use: literals, '.', '\x' escapes, '*' and '+' after an atom, a final '$'.
pub fn regexMatch(pattern: []const u8, module: []const u8) bool {
    return reAt(pattern, 0, module, 0);
}

const Atom = struct { len: usize, lit: ?u8 };

fn reAtom(pattern: []const u8, p: usize) Atom {
    if (pattern[p] == '\\' and p + 1 < pattern.len) return .{ .len = 2, .lit = pattern[p + 1] };
    if (pattern[p] == '.') return .{ .len = 1, .lit = null };
    return .{ .len = 1, .lit = pattern[p] };
}

fn reOne(lit: ?u8, c: u8) bool {
    return if (lit) |l| l == c else true;
}

fn reAt(pattern: []const u8, p: usize, module: []const u8, t: usize) bool {
    if (p == pattern.len) return true;
    if (pattern[p] == '$' and p + 1 == pattern.len) return t == module.len;
    const atom = reAtom(pattern, p);
    const next = p + atom.len;
    if (next < pattern.len and (pattern[next] == '*' or pattern[next] == '+')) {
        const least: usize = if (pattern[next] == '+') 1 else 0;
        var n: usize = 0;
        while (t + n < module.len and reOne(atom.lit, module[t + n])) n += 1;
        var k: usize = n + 1;
        while (k > least) {
            k -= 1;
            if (reAt(pattern, next + 1, module, t + k)) return true;
        }
        return false;
    }
    if (t < module.len and reOne(atom.lit, module[t])) return reAt(pattern, next, module, t + 1);
    return false;
}

/// One module as stored: qweight int32 [k / 8, n], scales f16 [k / 128, n], qzeros int32 [k / 128, n / 8].
pub const Host = struct { qweight: Tensor, scales: Tensor, qzeros: Tensor, n: usize, k: usize };

/// `module`'s tensors from `ck`, each refusal logged; the byte views live as long as the checkpoint's map.
pub fn read(ck: *Checkpoint, module: []const u8, n: usize, k: usize) !Host {
    var why: Why = .{};
    return readWhy(ck, module, n, k, &why) catch |e| {
        if (why.len > 0) std.log.err("{s}", .{why.text()});
        return e;
    };
}

/// `read`, the refusal kept in `why` instead of the log (tests read it).
pub fn readWhy(ck: *Checkpoint, module: []const u8, n: usize, k: usize, why: *Why) !Host {
    if (n == 0 or k == 0 or n % 8 != 0 or k % group != 0) {
        why.set("{s}: [n, k] of [{d}, {d}] is not 4-bit groups of 128 in words of 8 codes", .{ module, n, k });
        return error.UnexpectedTensor;
    }
    const qweight = try fetch(ck, module, "qweight", .i32, &.{ k / 8, n }, why);
    const scales = try fetch(ck, module, "scales", .f16, &.{ k / group, n }, why);
    const qzeros = try fetch(ck, module, "qzeros", .i32, &.{ k / group, n / 8 }, why);
    if (zerosOff(qzeros.bytes)) |at| {
        why.set("{s}.qzeros: byte {d} is 0x{x:0>2}, a symmetric weight's every nibble is 7", .{ module, at, qzeros.bytes[at] });
        return error.UnsupportedQuantization;
    }
    var buf: [320]u8 = undefined;
    const gidx = try std.fmt.bufPrint(&buf, "{s}.g_idx", .{module});
    if (ck.has(gidx)) {
        const g = try ck.get(gidx);
        if (g.numel() != 0) {
            why.set("{s}.g_idx: {t} {any}, act order is not read", .{ module, g.dtype, g.shape[0..g.rank] });
            return error.UnexpectedTensor;
        }
    }
    return .{ .qweight = qweight, .scales = scales, .qzeros = qzeros, .n = n, .k = k };
}

/// `{module}.{tail}` of `dtype` and `shape`, its bytes whole; a missing tensor refused by name.
fn fetch(ck: *Checkpoint, module: []const u8, tail: []const u8, dtype: ckpt.DType, shape: []const usize, why: *Why) !Tensor {
    var buf: [320]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "{s}.{s}", .{ module, tail });
    if (!ck.has(name)) {
        why.set("{s}: no {s}, a GPTQ weight stores it", .{ module, tail });
        return error.MissingTensor;
    }
    const t = try ck.get(name);
    var need: usize = 1;
    for (shape) |d| need = std.math.mul(usize, need, d) catch return error.UnexpectedTensor;
    if (t.is(dtype, shape)) {
        if (need * dtype.size() == t.bytes.len) return t;
        why.set("{s}: {d} bytes, {t} {any} needs {d}", .{ name, t.bytes.len, dtype, shape, need * dtype.size() });
        return error.UnexpectedTensor;
    }
    why.set("{s}: {t} {any}, expected {t} {any}", .{ name, t.dtype, t.shape[0..t.rank], dtype, shape });
    return error.UnexpectedTensor;
}

/// Weights of a module: its columns by its inputs.
pub fn elements(h: Host) usize {
    return h.n * h.k;
}

/// The dequantized weight of input `k` in column `n`: (code - 8) * scale, exact in binary64.
pub fn weight(h: Host, n: usize, k: usize) f64 {
    return @as(f64, @floatFromInt(@as(i32, code(h, n, k)) - 8)) * scale(h, n, k);
}

/// The code of input `k` in column `n`: nibble `k % 8` of the word at [k / 8, n] (0..15).
pub fn code(h: Host, n: usize, k: usize) u8 {
    std.debug.assert(n < h.n and k < h.k);
    const at = ((k / 8) * h.n + n) * 4;
    const w = std.mem.readInt(u32, h.qweight.bytes[at..][0..4], .little);
    return @intCast((w >> @intCast(4 * (k % 8))) & 0xF);
}

/// The fp16 scale of the group holding input `k` in column `n`, widened exactly.
pub fn scale(h: Host, n: usize, k: usize) f64 {
    std.debug.assert(n < h.n and k < h.k);
    const at = ((k / group) * h.n + n) * 2;
    const half: f16 = @bitCast(std.mem.readInt(u16, h.scales.bytes[at..][0..2], .little));
    return @floatCast(half);
}

/// Every weight, row-major [n, k], into `out`.
pub fn reference(h: Host, out: []f64) error{UnexpectedTensor}!void {
    if (!sized(h) or out.len != elements(h)) return error.UnexpectedTensor;
    for (0..h.n) |n| for (0..h.k) |k| {
        out[n * h.k + k] = weight(h, n, k);
    };
}

/// The stored tensors hold exactly what `Host`'s [n, k] says they do.
fn sized(h: Host) bool {
    if (h.n == 0 or h.k == 0 or h.n % 8 != 0 or h.k % group != 0) return false;
    const words = h.k / 8 * h.n;
    if (!h.qweight.is(.i32, &.{ h.k / 8, h.n }) or h.qweight.bytes.len != words * 4) return false;
    const scales = h.k / group * h.n;
    if (!h.scales.is(.f16, &.{ h.k / group, h.n }) or h.scales.bytes.len != scales * 2) return false;
    return h.qzeros.is(.i32, &.{ h.k / group, h.n / 8 }) and h.qzeros.bytes.len == scales / 8 * 4;
}

/// The first qzeros byte whose two nibbles are not both `zero_nibble`, else null.
pub fn zerosOff(qzeros: []const u8) ?usize {
    const all: u8 = zero_nibble << 4 | zero_nibble;
    for (qzeros, 0..) |b, i| if (b != all) return i;
    return null;
}

/// Every nibble of `qzeros` is `zero_nibble`, so every zero point is 8.
pub fn zerosSymmetric(qzeros: []const u8) bool {
    return zerosOff(qzeros) == null;
}

fn flag(v: ?std.json.Value) ?bool {
    const x = v orelse return null;
    return if (x == .bool) x.bool else null;
}

fn holds(v: ?std.json.Value, want: bool) bool {
    return (flag(v) orelse false) == want;
}

test {
    _ = @import("gptq_test.zig");
}
