//! Exact temporary BF16 views of a filled FP8 KV cache; source storage and decode remain untouched.
const std = @import("std");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Buffer = @import("memory.zig").DeviceBuffer;
const Stream = @import("stream.zig").Stream;
const launch = @import("launch.zig");
const fp8_attention = @import("fp8_attention_prompt.zig");
pub const Input = fp8_attention.Input;
pub const max_filled: u32 = 16384;
pub const max_kv_heads: u32 = 4;
pub const dimension: usize = 256;
pub const scratch_bytes: usize = 2 * max_kv_heads * max_filled * dimension * 2;
pub const symbol: [:0]const u8 = "tf_qwen_expand_kv";
pub const attention_shared: u32 = 98304;

pub fn selected(raw: ?[]const u8) !bool {
    const value = raw orelse return false;
    if (std.mem.eql(u8, value, "0")) return false;
    if (std.mem.eql(u8, value, "1")) return true;
    return error.InvalidExpandKVFlag;
}
pub fn requested() !bool {
    return selected(try @import("sys").environment.given("TF_QWEN_EXPAND_KV"));
}

pub const Expanded = struct {
    d: *const Driver,
    expand: Function,
    attention: Function,
    scratch: Buffer,
    last_stream: ?Stream = null,

    /// The engine owns the module and BF16 function; only the temporary buffer belongs to this value.
    pub fn borrow(module: Module, attention: Function) !Expanded {
        const expand = try module.function(symbol);
        return .{ .d = module.d, .expand = expand, .attention = attention, .scratch = try Buffer.alloc(module.d, scratch_bytes) };
    }
    pub fn deinit(e: *Expanded) void {
        if (e.last_stream) |s| @import("completion.zig").drain(e.d, s, "expanded KV attention did not drain");
        e.scratch.free();
        e.* = undefined;
    }
    pub fn eligible(p: Input) !bool {
        try p.validateAddressing();
        return p.shape.capacity <= fp8_attention.max_optimized_capacity and
            p.shape.prefix + p.shape.rows <= max_filled and p.shape.kv_heads <= max_kv_heads;
    }
    pub fn run(e: *Expanded, stream: Stream, p: Input) !bool {
        if (!try eligible(p)) {
            try e.validateRanges(p);
            return false;
        }
        try e.validate(p);
        if (e.last_stream) |last| if (last.handle != stream.handle or last.d != stream.d) {
            if (last.d != stream.d or !try last.done()) return error.ExpandKVBusy;
        };
        e.last_stream = stream;
        const s = p.shape;
        const length = s.prefix + s.rows;
        var expand: launch.Args = .{};
        for ([_]u64{ p.k, p.v, p.k_scales, p.v_scales, e.scratch.ptr }) |address| expand.add(address);
        for ([_]u32{ length, s.kv_heads, s.capacity }) |value| expand.add(@as(i32, @intCast(value)));
        expand.add(p.pages);
        expand.add(@as(i64, @intCast(s.page_bytes)));
        expand.add(@as(i32, @intCast(s.page_tokens)));
        const vectors = @as(usize, 2) * s.kv_heads * length * (dimension / 8);
        try launch.launch(e.expand, .{ .grid = .{ .x = @intCast((vectors + 255) / 256) }, .block = .{ .x = 256 } }, stream, &expand);
        var attention: launch.Args = .{};
        const values = e.scratch.ptr + @as(u64, s.kv_heads) * length * dimension * 2;
        for ([_]u64{ p.q, e.scratch.ptr, values, p.output }) |address| attention.add(address);
        for ([_]u32{ s.prefix, s.rows, s.heads, s.kv_heads, s.heads / s.kv_heads }) |value|
            attention.add(@as(i32, @intCast(value)));
        attention.add(s.scale);
        attention.add(@as(i32, @intCast(length)));
        try launch.launch(e.attention, .{
            .grid = .{ .x = (s.rows + 63) / 64, .y = s.heads / 2 },
            .block = .{ .x = 256 },
            .shared = attention_shared,
        }, stream, &attention);
        return true;
    }

    /// Page IDs and the backing cache remain owned and validated by the existing cache manager.
    pub fn validate(e: *const Expanded, p: Input) !void {
        try p.validate();
        const s = p.shape;
        const length = try std.math.add(u32, s.prefix, s.rows);
        const need = try std.math.mul(usize, @as(usize, 2) * s.kv_heads * dimension * 2, length);
        if (need > e.scratch.len) return error.InsufficientExpandKVScratch;
        try e.validateRanges(p);
    }

    fn validateRanges(e: *const Expanded, p: Input) !void {
        const s = p.shape;
        if (p.q % 16 != 0 or e.scratch.ptr % 16 != 0) return error.InvalidExpandKVPointer;
        const output_bytes = try std.math.mul(usize, @as(usize, s.rows) * s.heads, dimension * 2);
        const output = try Span.init(p.output, output_bytes);
        const scratch = try Span.init(e.scratch.ptr, e.scratch.len);
        if (output.overlaps(scratch)) return error.ExpandKVAlias;
        const pages = if (p.pages == 0) @as(usize, 1) else (@as(usize, s.capacity) + s.page_tokens - 1) / s.page_tokens;
        const rows = @as(usize, s.kv_heads) * (if (p.pages == 0) s.capacity else s.page_tokens);
        const stride = if (p.pages == 0) @as(usize, 0) else try std.math.mul(usize, pages - 1, s.page_bytes);
        const codes = try std.math.add(usize, stride, try std.math.mul(usize, rows, dimension));
        const scales = try std.math.add(usize, stride, try std.math.mul(usize, rows, 4));
        const reads = [_]Span{
            try Span.init(p.q, output_bytes),  try Span.init(p.k, codes),         try Span.init(p.v, codes),
            try Span.init(p.k_scales, scales), try Span.init(p.v_scales, scales),
        };
        for (reads) |read| if (output.overlaps(read) or scratch.overlaps(read)) return error.ExpandKVAlias;
        if (p.pages != 0) {
            const table = try Span.init(p.pages, try std.math.mul(usize, pages, 4));
            if (output.overlaps(table) or scratch.overlaps(table)) return error.ExpandKVAlias;
        }
    }
};

const Span = struct {
    begin: u64,
    end: u64,
    fn init(address: u64, count: usize) !Span {
        if (address == 0) return error.InvalidExpandKVPointer;
        return .{ .begin = address, .end = try std.math.add(u64, address, count) };
    }
    fn overlaps(a: Span, b: Span) bool {
        return a.begin < b.end and b.begin < a.end;
    }
};

test {
    _ = @import("kv_expand_test.zig");
}
