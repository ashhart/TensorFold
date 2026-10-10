//! The products the affine groups run every registered kernel on: the served models' projections and ragged sizes.

const registry = @import("core").registry;

/// A dense product of `rows` rows of x, or a routed one: `rows` tokens of `slots` experts each over stacked tables.
pub const Case = struct {
    name: []const u8,
    path: registry.Path = .decode,
    rows: usize = 1,
    n: usize,
    k: usize,
    bits: u8 = 4,
    group: u16 = 64,
    experts: usize = 0,
    slots: usize = 9,
    /// The routed down product: x has a row a pair, else a row a token.
    down: bool = false,
    /// The most rows an item of the plan holds.
    item_rows: usize = 8,
    /// Every expert takes the same number of pairs.
    even: bool = false,

    pub fn routed(c: Case) bool {
        return c.experts > 0;
    }
};

/// Decode: a lane round's few rows, dense and routed, on the engine's shapes, reaching every decode tile.
pub const decode = [_]Case{
    .{ .name = "35b lm_head", .n = 248320, .k = 2048 },
    .{ .name = "35b qkv", .n = 8192, .k = 2048 },
    .{ .name = "35b o", .n = 2048, .k = 4096 },
    .{ .name = "35b shared gate_up", .n = 1024, .k = 2048 },
    .{ .name = "35b shared down", .n = 2048, .k = 512 },
    .{ .name = "35b kv", .n = 512, .k = 2048 },
    .{ .name = "35b routed gate_up", .n = 1024, .k = 2048, .experts = 256 },
    .{ .name = "35b routed down", .n = 2048, .k = 512, .experts = 256, .down = true },
    .{ .name = "9b qkv", .n = 8192, .k = 4096 },
    .{ .name = "9b gate_up", .n = 24576, .k = 4096 },
    .{ .name = "9b down", .n = 4096, .k = 12288 },
    .{ .name = "9b lm_head", .n = 248320, .k = 4096 },
    .{ .name = "9b down 6-bit", .n = 4096, .k = 12288, .bits = 6 },
    .{ .name = "9b down 8-bit", .n = 4096, .k = 12288, .bits = 8 },
    .{ .name = "9b gate_up 6-bit", .n = 24576, .k = 4096, .bits = 6 },
    .{ .name = "9b gate_up 8-bit", .n = 24576, .k = 4096, .bits = 8 },
    .{ .name = "9b8 qkv", .n = 12352, .k = 4096, .bits = 8 },
    .{ .name = "9b8 o", .n = 4096, .k = 4096, .bits = 8 },
    .{ .name = "9b8 lm_head", .n = 248320, .k = 4096, .bits = 8 },
    .{ .name = "9b8 kv", .n = 1024, .k = 4096, .bits = 8 },
    .{ .name = "2-bit", .n = 8192, .k = 2048, .bits = 2 },
    .{ .name = "3-bit", .n = 8192, .k = 2048, .bits = 3 },
    .{ .name = "5-bit", .n = 8192, .k = 2048, .bits = 5 },
    .{ .name = "group 32", .n = 8192, .k = 2048, .group = 32 },
    .{ .name = "group 128", .n = 8192, .k = 2048, .group = 128 },
    .{ .name = "ragged", .n = 1001, .k = 2112 },
    .{ .name = "stride k4032", .n = 24576, .k = 4032 },
    .{ .name = "stride k4160", .n = 24576, .k = 4160 },
    .{ .name = "stride k2048", .n = 49152, .k = 2048 },
    .{ .name = "stride k8192", .n = 12288, .k = 8192 },
    .{ .name = "35b qkv x2", .rows = 2, .n = 8192, .k = 2048 },
    .{ .name = "35b qkv x4", .rows = 4, .n = 8192, .k = 2048 },
    .{ .name = "35b qkv x8", .rows = 8, .n = 8192, .k = 2048 },
    .{ .name = "35b qkv x16", .rows = 16, .n = 8192, .k = 2048 },
    .{ .name = "9b gate_up x4", .rows = 4, .n = 24576, .k = 4096 },
    .{ .name = "9b gate_up x8", .rows = 8, .n = 24576, .k = 4096 },
    .{ .name = "9b gate_up x12", .rows = 12, .n = 24576, .k = 4096 },
    .{ .name = "9b gate_up x16", .rows = 16, .n = 24576, .k = 4096 },
    .{ .name = "35b routed gate_up x2", .rows = 2, .n = 1024, .k = 2048, .experts = 256 },
    .{ .name = "35b routed gate_up x4", .rows = 4, .n = 1024, .k = 2048, .experts = 256 },
    .{ .name = "35b routed gate_up x8", .rows = 8, .n = 1024, .k = 2048, .experts = 256 },
    .{ .name = "35b routed down x4", .rows = 4, .n = 2048, .k = 512, .experts = 256, .down = true },
};

// Prefill: a prompt's rows. The routed rows are tokens, 8 experts each over 256, in items of at most 128 rows.
pub const prefill = [_]Case{
    .{ .name = "35b qkv", .path = .prefill, .rows = 2048, .n = 8192, .k = 2048 },
    .{ .name = "35b qkv", .path = .prefill, .rows = 8192, .n = 8192, .k = 2048 },
    .{ .name = "35b o", .path = .prefill, .rows = 2048, .n = 2048, .k = 4096 },
    .{ .name = "35b o", .path = .prefill, .rows = 8192, .n = 2048, .k = 4096 },
    .{ .name = "35b shared gate_up", .path = .prefill, .rows = 8192, .n = 1024, .k = 2048 },
    .{ .name = "35b expert gate_up", .path = .prefill, .rows = 64, .n = 1024, .k = 2048 },
    .{ .name = "35b expert gate_up", .path = .prefill, .rows = 256, .n = 1024, .k = 2048 },
    .{ .name = "35b expert down", .path = .prefill, .rows = 256, .n = 2048, .k = 512 },
    .{ .name = "35b routed gate_up", .path = .prefill, .rows = 2048, .n = 1024, .k = 2048, .experts = 256, .slots = 8, .item_rows = 128 },
    .{ .name = "35b routed gate_up", .path = .prefill, .rows = 8192, .n = 1024, .k = 2048, .experts = 256, .slots = 8, .item_rows = 128 },
    .{ .name = "35b routed even 32", .path = .prefill, .rows = 1024, .n = 1024, .k = 2048, .experts = 256, .slots = 8, .item_rows = 128, .even = true },
    .{ .name = "35b routed even 64", .path = .prefill, .rows = 2048, .n = 1024, .k = 2048, .experts = 256, .slots = 8, .item_rows = 128, .even = true },
    .{ .name = "35b routed down", .path = .prefill, .rows = 2048, .n = 2048, .k = 512, .experts = 256, .slots = 8, .item_rows = 128, .down = true },
    .{ .name = "35b routed down", .path = .prefill, .rows = 8192, .n = 2048, .k = 512, .experts = 256, .slots = 8, .item_rows = 128, .down = true },
    .{ .name = "9b qkv", .path = .prefill, .rows = 2048, .n = 8192, .k = 4096 },
    .{ .name = "9b gate_up", .path = .prefill, .rows = 2048, .n = 24576, .k = 4096 },
    .{ .name = "9b gate_up", .path = .prefill, .rows = 8192, .n = 12288, .k = 4096 },
    .{ .name = "9b down", .path = .prefill, .rows = 2048, .n = 4096, .k = 12288 },
    .{ .name = "9b down", .path = .prefill, .rows = 8192, .n = 4096, .k = 12288 },
    .{ .name = "27b gate_up", .path = .prefill, .rows = 2048, .n = 17408, .k = 5120 },
    .{ .name = "27b down", .path = .prefill, .rows = 2048, .n = 5120, .k = 17408 },
    .{ .name = "27b q", .path = .prefill, .rows = 8192, .n = 12288, .k = 5120 },
    .{ .name = "vocab", .path = .prefill, .rows = 128, .n = 248320, .k = 2048 },
    .{ .name = "3-bit", .path = .prefill, .rows = 2048, .n = 4096, .k = 2048, .bits = 3 },
    .{ .name = "6-bit", .path = .prefill, .rows = 2048, .n = 4096, .k = 2048, .bits = 6 },
    .{ .name = "8-bit", .path = .prefill, .rows = 2048, .n = 4096, .k = 2048, .bits = 8 },
    .{ .name = "group 32", .path = .prefill, .rows = 2048, .n = 4096, .k = 2048, .group = 32 },
    .{ .name = "group 128", .path = .prefill, .rows = 2048, .n = 4096, .k = 2048, .group = 128 },
    .{ .name = "edges", .path = .prefill, .rows = 1000, .n = 1001, .k = 2112 },
};

pub const widths = [_]u8{ 2, 3, 4, 5, 6, 8 };
pub const groups = [_]u16{ 32, 64, 128 };

/// A short prompt's few rows, for the timings of each tile on dense and routed products.
pub const short_rows = [_]usize{ 1, 2, 4, 8, 13, 24, 32, 64 };
pub const short_tokens = [_]usize{ 1, 13, 32, 64, 128, 256 };
pub const short = [_]Case{
    .{ .name = "9b qkv", .path = .prefill, .n = 8192, .k = 4096 },
    .{ .name = "9b gate_up", .path = .prefill, .n = 24576, .k = 4096 },
    .{ .name = "9b down", .path = .prefill, .n = 4096, .k = 12288 },
    .{ .name = "35b o", .path = .prefill, .n = 2048, .k = 4096 },
    .{ .name = "35b routed gate_up", .path = .prefill, .n = 1024, .k = 2048, .experts = 256, .slots = 8, .item_rows = 128 },
    .{ .name = "35b routed down", .path = .prefill, .n = 2048, .k = 512, .experts = 256, .slots = 8, .item_rows = 128, .down = true },
};
