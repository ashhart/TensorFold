//! What a Qwen3.5 text tower must declare for the kernels, as named rows; each backend's column says which it reads.
const std = @import("std");

/// One word per refusal, the same on every backend.
pub const Error = error{ UnsupportedActivation, UnsupportedAttentionGate, UnsupportedStateType };

/// The config entry points that read the rows: CUDA, Metal's 27B (`qwen27`), Metal's 2B, HIP.
pub const Backend = enum { cuda, metal27, metal2b, hip };

/// How a backend reads a row: not at all, when present, or refusing a tower without it.
pub const Read = enum { skip, optional, required };

/// The value the kernels compute: a string, or a JSON boolean.
pub const Want = union(enum) {
    text: []const u8,
    flag: bool,

    /// `v` is the wanted value; `numeric` also takes the 0 or 1 some conversions write for a boolean.
    pub fn holds(w: Want, v: std.json.Value, numeric: bool) bool {
        return switch (w) {
            .text => |s| v == .string and std.mem.eql(u8, v.string, s),
            .flag => |b| (v == .bool and v.bool == b) or (numeric and v == .integer and v.integer == @intFromBool(b)),
        };
    }
};

/// Each backend's reading of one row.
pub const Columns = struct { cuda: Read, metal27: Read, metal2b: Read, hip: Read };

/// A key, the value the kernels compute, the refusal when the tower declares another, and who reads it.
pub const Row = struct {
    key: []const u8,
    want: Want,
    refusal: Error,
    read: Columns,

    pub fn by(r: Row, b: Backend) Read {
        return switch (b) {
            inline else => |t| @field(r.read, @tagName(t)),
        };
    }
};

/// The table: a backend's `skip` is a row it does not check today (listed in #2460's config survey).
pub const rows = [_]Row{
    .{
        .key = "hidden_act",
        .want = .{ .text = "silu" },
        .refusal = error.UnsupportedActivation,
        .read = .{ .cuda = .required, .metal27 = .optional, .metal2b = .required, .hip = .optional },
    },
    .{
        .key = "output_gate_type", // "swish": att * sigmoid(gate), as the kernels apply it
        .want = .{ .text = "swish" },
        .refusal = error.UnsupportedAttentionGate,
        .read = .{ .cuda = .skip, .metal27 = .optional, .metal2b = .skip, .hip = .optional },
    },
    .{
        .key = "attn_output_gate",
        .want = .{ .flag = true },
        .refusal = error.UnsupportedAttentionGate,
        .read = .{ .cuda = .required, .metal27 = .optional, .metal2b = .required, .hip = .optional },
    },
    .{
        .key = "attention_bias",
        .want = .{ .flag = false },
        .refusal = error.UnsupportedAttentionGate,
        .read = .{ .cuda = .required, .metal27 = .skip, .metal2b = .required, .hip = .optional },
    },
    .{
        .key = "linear_attn_state_dtype",
        .want = .{ .text = "float32" },
        .refusal = error.UnsupportedStateType,
        .read = .{ .cuda = .skip, .metal27 = .optional, .metal2b = .skip, .hip = .optional },
    },
    .{
        .key = "mamba_ssm_dtype",
        .want = .{ .text = "float32" },
        .refusal = error.UnsupportedStateType,
        .read = .{ .cuda = .skip, .metal27 = .skip, .metal2b = .required, .hip = .optional },
    },
};

/// Only HIP's reader takes 0 and 1 for a boolean row.
pub fn numericFlags(b: Backend) bool {
    return b == .hip;
}

/// Refuses `tower` by the first row `b` reads whose value is absent (when required) or not the wanted one.
pub fn check(tower: std.json.ObjectMap, b: Backend) Error!void {
    for (rows) |r| {
        const read = r.by(b);
        if (read == .skip) continue;
        const v = tower.get(r.key) orelse {
            if (read == .required) return r.refusal;
            continue;
        };
        if (!r.want.holds(v, numericFlags(b))) return r.refusal;
    }
}

fn parsed(a: std.mem.Allocator, text: []const u8) !std.json.ObjectMap {
    return (try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{})).object;
}

/// A tower holding every row's wanted value.
const full =
    \\{"hidden_act":"silu","output_gate_type":"swish","attn_output_gate":true,"attention_bias":false,
    \\ "linear_attn_state_dtype":"float32","mamba_ssm_dtype":"float32"}
;

test "qwen35 rules: every backend admits the wanted values, and refuses each row it reads by the row's word" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (std.enums.values(Backend)) |b| {
        try check(try parsed(a, full), b);
        for (rows) |r| {
            var tower = try parsed(a, full);
            try tower.put(a, r.key, switch (r.want) {
                .text => .{ .string = "other" },
                .flag => |f| .{ .bool = !f },
            });
            const got = check(tower, b);
            if (r.by(b) == .skip) try got else try std.testing.expectError(r.refusal, got);
            _ = tower.orderedRemove(r.key);
            const absent = check(tower, b);
            if (r.by(b) == .required) try std.testing.expectError(r.refusal, absent) else try absent;
        }
    }
}

test "qwen35 rules: an empty tower passes where no row is required; 0 and 1 count as flags on HIP only" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try check(try parsed(a, "{}"), .hip);
    try check(try parsed(a, "{}"), .metal27);
    try std.testing.expectError(error.UnsupportedActivation, check(try parsed(a, "{}"), .cuda));
    try std.testing.expectError(error.UnsupportedActivation, check(try parsed(a, "{}"), .metal2b));
    const numeric = "{\"attn_output_gate\":1,\"attention_bias\":0}";
    try check(try parsed(a, numeric), .hip);
    try std.testing.expectError(error.UnsupportedAttentionGate, check(try parsed(a, numeric), .metal27));
    const null_gate = "{\"attn_output_gate\":null}";
    try std.testing.expectError(error.UnsupportedAttentionGate, check(try parsed(a, null_gate), .hip));
}
