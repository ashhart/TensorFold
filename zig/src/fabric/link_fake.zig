//! Both mcdma-rpcd daemons of one link, played over the fake fabric: rank 0 is the connect end, rank 1 the listen end.
const std = @import("std");
const layout = @import("layout.zig");
const words = @import("words.zig");
const mailbox = @import("mailbox.zig");
const fake = @import("fake.zig");

/// One link: each rank's window is its mailbox, and `step` moves words and payloads as rpcd_connect.c and rpcd_listen.c do.
pub const Link = struct {
    cluster: *fake.Cluster,
    req: usize,
    rep: usize,
    direct: bool,
    generation: u64 = 0,
    last_request: u32 = 0,
    last_staged: u32 = 0,
    last_ready: u32 = 0,

    pub fn init(gpa: std.mem.Allocator, req: usize, rep: usize, direct: bool, mode: fake.Mode) !Link {
        const c = try fake.Cluster.init(gpa, 2, req + rep, mode, null);
        for (c.windows) |w| {
            std.mem.writeInt(u64, w[layout.sizes..][0..8], req, .little);
            std.mem.writeInt(u64, w[layout.sizes + 8 ..][0..8], rep, .little);
        }
        return .{ .cluster = c, .req = req, .rep = rep, .direct = direct };
    }

    pub fn deinit(l: *Link) void {
        l.cluster.deinit();
    }

    pub fn box(l: *Link, rank: u32) mailbox.Mailbox {
        return mailbox.Mailbox.fromMemory(l.cluster.windows[rank]) catch unreachable;
    }

    fn word(l: *Link, rank: u32, offset: usize) *u64 {
        return @ptrCast(@alignCast(l.cluster.windows[rank].ptr + offset));
    }

    /// HELLO and READY's effect: both ends' words cleared, a new generation, then link up.
    pub fn up(l: *Link) void {
        for ([_]usize{ layout.request_word, l.req + layout.ready_word, l.req + layout.done_word }) |o| words.native.store(l.word(0, o), 0);
        for ([_]usize{ layout.request_word, l.req + layout.staged_word }) |o| words.native.store(l.word(1, o), 0);
        l.last_request = 0;
        l.last_staged = 0;
        l.last_ready = 0;
        l.generation += 1;
        words.native.store(l.word(0, layout.generation), l.generation);
        words.native.store(l.word(0, layout.link_up), 1);
    }

    /// The link drops: in-flight calls are lost and the connect end says so.
    pub fn down(l: *Link) void {
        words.native.store(l.word(0, layout.link_up), 0);
    }

    /// One pass of both daemons' loops; true when anything moved.
    pub fn step(l: *Link) !bool {
        if (words.load(l.word(0, layout.link_up)) != 1) return false;
        const mac = l.cluster.endpoint(0);
        const peer = l.cluster.endpoint(1);
        var moved = false;
        const w = words.load(l.word(0, layout.request_word));
        if (layout.seqOf(w) != 0 and layout.seqOf(w) != l.last_request) {
            l.last_request = layout.seqOf(w);
            const len = layout.lenOf(w);
            try mac.write(1, layout.ctrl, l.cluster.windows[0][layout.ctrl..][0..len]);
            try mac.signal(1, layout.request_word, w);
            moved = true;
        }
        const s = words.load(l.word(1, l.req + layout.staged_word));
        if (layout.seqOf(s) != 0 and layout.seqOf(s) != l.last_staged) {
            l.last_staged = layout.seqOf(s);
            if (l.direct) {
                try peer.write(0, l.req + layout.ctrl, l.cluster.windows[1][l.req + layout.ctrl ..][0..layout.lenOf(s)]);
                try peer.signal(0, l.req + layout.done_word, s);
            } else {
                try peer.signal(0, l.req + layout.ready_word, s);
            }
            moved = true;
        }
        const r = words.load(l.word(0, l.req + layout.ready_word));
        if (!l.direct and layout.seqOf(r) != 0 and layout.seqOf(r) != l.last_ready) {
            l.last_ready = layout.seqOf(r);
            try mac.read(1, l.req + layout.ctrl, l.cluster.windows[0][l.req + layout.ctrl ..][0..layout.lenOf(r)]);
            words.native.store(l.word(0, l.req + layout.done_word), r);
            moved = true;
        }
        return moved;
    }

    /// Run both daemons until `stop` is set.
    pub fn run(l: *Link, stop: *const std.atomic.Value(bool)) void {
        while (!stop.load(.acquire)) {
            const moved = l.step() catch return;
            if (!moved) std.atomic.spinLoopHint();
        }
    }
};
