//! Qwen3-ASR on Metal: the shared tower and Qwen3 schedules over this directory's bindings, one device.
const std = @import("std");
const core = @import("core");
const cfg = @import("../config.zig");
const mel = @import("../mel.zig");
const tower = @import("../tower.zig");
const lm = @import("../lm.zig");
const gpu = @import("gpu.zig");
const mtower = @import("tower.zig");
const mlm = @import("lm.zig");

pub const Gpu = gpu.Gpu;
pub const Dump = mtower.Dump;
pub const Splice = lm.Splice;

pub const Options = struct {
    group: usize = 16, // chunks a convolution pass
    audio_rows: usize, // the longest clip's tower rows
    pass_rows: usize, // the most ids a language model pass takes
    positions: usize, // the most positions the caches hold
};

/// The bf16 bytes of a checkpoint's tensors by name and shape, for tower.Weights.load and lm.Weights.load.
const Source = struct {
    ck: *core.Checkpoint,

    pub fn expect(s: Source, name: []const u8, shape: []const usize) ![]const u8 {
        return (try s.ck.expect(name, .bf16, shape)).bytes;
    }
};

/// The device as the schedules' store: `put` and `taps` place tensors, `alloc` hands out scratch.
const Store = struct {
    g: *gpu.Gpu,

    pub fn alloc(st: Store, n: usize) !u64 {
        return st.g.alloc(n);
    }

    pub fn put(st: Store, bytes: []const u8) !u64 {
        const p = try st.g.alloc(bytes.len);
        @memcpy(st.g.host(p, bytes.len), bytes);
        return p;
    }

    /// A conv's [co, c 9] taps as stored: q3a_conv_cols writes its columns in the same order.
    pub fn taps(st: Store, bytes: []const u8, _: usize, _: usize) !u64 {
        return st.put(bytes);
    }
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    c: cfg.Config,
    g: *gpu.Gpu,
    weight_bytes: u64,
    tw: tower.Weights,
    lw: lm.Weights,
    tb: tower.Buffers,
    lb: lm.Buffers,
    tr: mtower.Tower,
    lr: mlm.Lm,

    /// The checkpoint at `dir` on the device (every tensor used), with scratch for `o`'s clips and passes.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, metallib: []const u8, dir: []const u8, c: cfg.Config, o: Options) !*Model {
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.gpa = gpa;
        m.c = c;
        m.g = try gpu.Gpu.open(gpa, metallib);
        errdefer m.g.close();
        const st: Store = .{ .g = m.g };
        {
            var ck = try core.Checkpoint.openModel(gpa, io, dir);
            defer ck.close();
            const src: Source = .{ .ck = &ck };
            m.tw = try tower.Weights.load(gpa, c.audio, src, st, o.group);
            errdefer m.tw.deinit(gpa);
            m.lw = try lm.Weights.load(gpa, c.text, src, st);
            errdefer m.lw.deinit(gpa);
            if (ck.unused() != 0) return error.UnexpectedTensors;
        }
        errdefer m.tw.deinit(gpa);
        errdefer m.lw.deinit(gpa);
        m.weight_bytes = m.g.bytes;
        m.tb = try tower.Buffers.alloc(c.audio, o.group, o.audio_rows, st);
        m.lb = try lm.Buffers.alloc(c.text, o.pass_rows, st);
        m.tr = try mtower.Tower.init(m.g, c.audio, o.group);
        m.lr = try mlm.Lm.init(gpa, m.g, c.text, o.pass_rows, o.positions);
        return m;
    }

    pub fn deinit(m: *Model) void {
        m.lr.deinit(m.gpa);
        m.lw.deinit(m.gpa);
        m.tw.deinit(m.gpa);
        m.g.close();
        m.gpa.destroy(m);
    }

    /// The tower over `feats` into `out` [rows, audio.out] bf16, each stage to `sink` when given; the rows returned.
    pub fn encode(m: *Model, feats: mel.Features, out: u64, sink: ?Dump) !usize {
        m.tr.sink = sink;
        defer m.tr.sink = null;
        return tower.encode(&m.tr, m.gpa, m.c.audio, &m.tw, m.tb, feats, out);
    }

    /// One language model pass over `ids` at [pos, pos + ids.len), `audio` spliced in: the greedy next id.
    pub fn forward(m: *Model, ids: []const u32, pos: usize, audio: ?Splice) !u32 {
        return m.forwardDump(ids, pos, audio, null);
    }

    /// forward with its stages (lm_embed, lm_layer{i}, lm_norm) to `sink`.
    pub fn forwardDump(m: *Model, ids: []const u32, pos: usize, audio: ?Splice, sink: ?Dump) !u32 {
        m.lr.sink = sink;
        defer m.lr.sink = null;
        return lm.forward(&m.lr, m.c.text, &m.lw, m.lb, ids, pos, audio);
    }
};
