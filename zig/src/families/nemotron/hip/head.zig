//! The MTP draft head on a lane: kept rows absorbed into its cache, then greedy one-row steps chained into drafts.
const std = @import("std");
const hip = @import("hip");
const engine = @import("engine.zig");
const forward = @import("forward.zig");
const pages = @import("pages.zig");

const Tensor = hip.ops.Tensor;

/// Drafts a chain holds at most.
pub const max_depth = 3;

/// Token i + 2's draft from hidden row i and token i + 1: drafts never change verified tokens, only their rounds.
pub const Head = struct {
    e: *engine.Engine,
    gpa: std.mem.Allocator,
    /// The draft rows' token ids on the host: a draft is the id of the head's best row.
    ids: []u32,
    /// A step's best row, on the device and read back.
    arg: hip.DeviceBuffer,
    host: hip.HostBuffer,
    /// The last step's final-normed row: the next chained step's hidden input.
    normed: hip.DeviceBuffer,
    /// Chained steps run, and drafts they gave.
    steps: u64 = 0,
    drafts: u64 = 0,

    /// The head over the engine's loaded MTP weights; null when the checkpoint has none (or it was not loaded).
    pub fn init(gpa: std.mem.Allocator, e: *engine.Engine) !?*Head {
        const mt = e.model_().mtp orelse return null;
        const h = try gpa.create(Head);
        errdefer gpa.destroy(h);
        const ids = try gpa.alloc(u32, mt.draft_count);
        errdefer gpa.free(ids);
        try hip.runtime.check(e.driver.api.hipMemcpyDtoH(@ptrCast(ids.ptr), @ptrFromInt(mt.draft_ids), ids.len * 4));
        var arg = try hip.DeviceBuffer.alloc(&e.driver, 4);
        errdefer arg.free();
        var host = try hip.HostBuffer.alloc(&e.driver, 4);
        errdefer host.free();
        var normed = try hip.DeviceBuffer.alloc(&e.driver, e.model_().dims.hidden * e.act.size());
        errdefer normed.free();
        h.* = .{ .e = e, .gpa = gpa, .ids = ids, .arg = arg, .host = host, .normed = normed };
        return h;
    }

    pub fn deinit(h: *Head) void {
        if (h.steps > 0) std.log.info("draft head: {d} chained steps, {d} drafts", .{ h.steps, h.drafts });
        h.normed.free();
        h.host.free();
        h.arg.free();
        h.gpa.free(h.ids);
        h.gpa.destroy(h);
    }

    /// Device bytes the head holds beside the weights.
    pub fn deviceBytes(e: *const engine.Engine) usize {
        return 4 + e.model_().dims.hidden * e.act.size();
    }

    /// The `n` pairs (row i of `hidden`, `next[i]`) absorbed at the stream's head positions (`prompt`: prompt rows).
    pub fn absorb(
        h: *Head,
        o: hip.ops.Ops,
        s: *pages.Stream,
        hidden: Tensor,
        next: []const u32,
        prompt: bool,
    ) !forward.Drafted {
        const e = h.e;
        const dev = try e.ids(o, next);
        const d = try forward.draft(
            o,
            e.model_(),
            e.mtp_slot,
            .{ .pool = &e.pool, .stream = s },
            dev,
            hidden,
            next.len,
            s.mtp_len,
            prompt,
        );
        s.mtp_len += next.len;
        return d;
    }

    /// The best draft row of the last of `rows` logits rows, as a token.
    fn best(h: *Head, o: hip.ops.Ops, logits: Tensor, rows: usize) !u32 {
        const width = h.e.model_().mtp.?.draft_count;
        try o.argmaxRows(forward.at(logits, (rows - 1) * width), 1, width, h.arg.base());
        try hip.raw.download(h.arg, 0, h.host.bytes[0..4], o.stream);
        try h.e.stream.synchronize();
        return h.ids[@intCast(h.host.slice(i32)[0])];
    }

    /// `next` absorbed after `hidden` (its rows), then `out.len` greedy drafts chained from the last pair: their count.
    pub fn chain(
        h: *Head,
        o: hip.ops.Ops,
        s: *pages.Stream,
        hidden: Tensor,
        next: []const u32,
        out: []u32,
        prompt: bool,
    ) !usize {
        const e = h.e;
        const m = e.model_();
        const d0 = try h.absorb(o, s, hidden, next, prompt);
        var logits = d0.logits;
        var normed = d0.normed;
        var rows = next.len;
        var pos = s.mtp_len;
        var got: usize = 0;
        while (got < out.len) {
            out[got] = try h.best(o, logits, rows);
            got += 1;
            h.drafts += 1;
            if (got == out.len) break;
            // the next step reads the last final-normed row and the draft, at a position the kept rows will overwrite
            const width = m.dims.hidden * e.act.size();
            try hip.raw.copy(h.normed, 0, normed.ptr + (rows - 1) * width, width, o.stream);
            const dev = try e.ids(o, out[got - 1 ..][0..1]);
            const d = try forward.draft(
                o,
                m,
                e.mtp_slot,
                .{ .pool = &e.pool, .stream = s },
                dev,
                .{ .ptr = h.normed.base(), .kind = e.act },
                1,
                pos,
                false,
            );
            h.steps += 1;
            logits = d.logits;
            normed = d.normed;
            rows = 1;
            pos += 1;
        }
        return got;
    }
};
