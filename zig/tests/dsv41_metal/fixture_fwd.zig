//! The tiny model's whole forward on this GPU: the rig, the forward's buffers, Engram row shards and the hash host.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");
const fixture = @import("fixture.zig");

const Host = host.engram_host.Host;
const er = host.engram_rows;

pub const Fwd = struct {
    a: std.mem.Allocator,
    r: *fixture.Rig,
    b: pkg.forward.Bufs,
    tokens: []u32,
    host: Host,
    rows: *er.Rows,
    tmp: std.testing.TmpDir,
    /// each Engram layer's shard bytes, for the reference's own row reads
    shards: [4][]u8 = undefined,

    /// The tiny forward model from `seed`, its Engram tables drawn from `seed` too, written as row shards.
    pub fn init(a: std.mem.Allocator, io: std.Io, limit: u32, seed: u64) !*Fwd {
        return initMode(a, io, limit, seed, .copy);
    }

    pub fn initMode(a: std.mem.Allocator, io: std.Io, limit: u32, seed: u64, mode: fixture.Rig.Mode) !*Fwd {
        return initWithMode(a, io, fixture.forwardConfig(), limit, seed, mode);
    }

    /// The same over another tiny config (Engram layer ids and taps as it names them).
    pub fn initWith(a: std.mem.Allocator, io: std.Io, cfg: host.config.Config, limit: u32, seed: u64) !*Fwd {
        return initWithMode(a, io, cfg, limit, seed, .copy);
    }

    pub fn initWithMode(a: std.mem.Allocator, io: std.Io, cfg: host.config.Config, limit: u32, seed: u64, mode: fixture.Rig.Mode) !*Fwd {
        const f = try a.create(Fwd);
        errdefer a.destroy(f);
        f.a = a;
        f.r = try fixture.Rig.initMode(a, io, cfg, limit, seed, mode);
        errdefer f.r.deinit();
        const c = &f.r.c;
        f.b = try pkg.forward.Bufs.init(f.r.device, c);
        errdefer f.b.deinit();
        f.tokens = try a.alloc(u32, c.vocab);
        errdefer a.free(f.tokens);
        for (f.tokens, 0..) |*t, i| t.* = @intCast(i);
        var prng = std.Random.DefaultPrng.init(seed ^ 0xe1);
        const rng = prng.random();
        f.host = .{ .layer_ids = c.engram_layers.items(), .max_ngram = c.engram_max_ngram, .heads = c.engram_heads, .token_map = f.tokens, .pad_id = c.engram_pad };
        for (0..c.engram_layers.len) |li| for (0..c.engram_max_ngram) |s| {
            f.host.mult[li][s] = @as(i64, rng.int(u32)) * 2 + 1;
        };
        f.host.layout(c.engram_vocab);
        const cols = c.hashCols();
        for (0..c.engram_layers.len) |li| {
            var total: u64 = 0;
            for (0..cols) |k| total += @intCast(f.host.primes[li][k]);
            c.engram_rows.buf[li] = total;
        }
        f.tmp = std.testing.tmpDir(.{});
        errdefer f.tmp.cleanup();
        for (c.engram_layers.items(), 0..) |l, li| f.shards[li] = try writeShard(a, io, f.tmp.dir, l, c.engram_rows.buf[li], rng);
        const dir = try f.tmp.dir.realPathFileAlloc(io, ".", a);
        defer a.free(dir);
        f.rows = try er.Rows.open(a, dir, 0, 1, c.engram_layers.items(), 4096);
        return f;
    }

    pub fn deinit(f: *Fwd) void {
        f.rows.close();
        f.tmp.cleanup();
        for (f.shards[0..f.r.c.engram_layers.len]) |b| f.a.free(b);
        f.a.free(f.tokens);
        f.b.deinit();
        f.r.deinit();
        f.a.destroy(f);
    }

    pub fn lookup(f: *Fwd) pkg.forward.Lookup {
        return .{ .a = f.a, .host = &f.host, .source = f.rows.source() };
    }

    pub fn model(f: *const Fwd) pkg.forward.Model {
        return .{ .top = &f.r.top.?, .layers = f.r.layers };
    }
};

/// A layer's whole table as one rank's shard: the packer's header, then 256 E4M3 bytes and 8 E8M0 scales a row.
fn writeShard(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, layer: u32, total: u64, rng: std.Random) ![]u8 {
    const n: usize = @intCast(er.header_bytes + total * er.row_bytes);
    const b = try a.alloc(u8, n);
    errdefer a.free(b);
    @memset(b, 0);
    for ([_]u64{ er.magic, layer, 0, total, total, er.row_bytes }, 0..) |v, i| std.mem.writeInt(u64, b[8 * i ..][0..8], v, .little);
    for (0..@intCast(total)) |k| {
        const rec = b[er.header_bytes + k * er.row_bytes ..][0..er.row_bytes];
        for (rec[0..256]) |*x| x.* = (rng.int(u8) & 0x87) | (@as(u8, 4 + rng.uintLessThan(u8, 5)) << 3);
        for (rec[256..264]) |*x| x.* = 125 + rng.uintLessThan(u8, 4);
    }
    var nb: [48]u8 = undefined;
    try dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&nb, "engram-l{d}-r0of1.bin", .{layer}), .data = b });
    return b;
}

/// The `n` ids after `lookback` drawn from `seed`.
pub fn ids(a: std.mem.Allocator, c: *const host.config.Config, n: usize, seed: u64) ![]u32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const out = try a.alloc(u32, n);
    for (out) |*v| v.* = prng.random().uintLessThan(u32, c.vocab);
    return out;
}

/// The `keep` ids before index `at` of a token history (the lookback of a launch starting there).
pub fn lookbackAt(hist: []const u32, at: usize, keep: usize) []const u32 {
    return hist[at -| keep..at];
}
