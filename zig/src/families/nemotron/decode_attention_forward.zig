//! Bind the forward's buffers to token-at-a-time chain attention.
const decode = @import("decode_attention.zig");

pub fn encode(f: anytype, e: anytype, kv: anytype, len: usize, first: usize, rows: usize) void {
    const s = f.s;
    var c = decode.Call{
        .head_dim = f.c.head_dim,
        .heads = f.c.heads,
        .kv_heads = f.c.kv_heads,
        .qkv_n = f.c.qkvDim(),
        .capacity = kv.capacity,
        .len = len,
        .first = first,
        .rows = rows,
        .qkv = .{ .buf = s.qkv },
        .qp = .{ .buf = s.qp },
        .k = .{ .buf = kv.k },
        .v = .{ .buf = kv.v },
        .po = .{ .buf = s.po },
        .pm = .{ .buf = s.pm },
        .pl = .{ .buf = s.pl },
        .att = .{ .buf = s.att },
        .out = .{ .buf = s.ax },
    };
    if (kv.paged) |p| {
        c.k = .{ .buf = p.arena(), .off = p.k_off };
        c.v = .{ .buf = p.arena(), .off = p.v_off };
        c.table = .{ .buf = p.table(), .off = p.tableOff() };
        c.page_bytes = p.pageBytes();
    }
    decode.encode(e, .{
        .q = f.k.get("tf_attn_q"),
        .partial = if (kv.paged != null) f.k.paged.?.partial[0] else f.k.get("attn_partial_1"),
        .merge = f.k.get("attn_merge"),
        .out = f.k.get("tf_attn_out"),
    }, c);
}
