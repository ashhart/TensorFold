//! The startup plan's windows past the engine's own: counted only where the rest past every fixed part holds them.
const std = @import("std");
const kv = @import("kv_pool.zig");
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

const mib: u64 = 1 << 20;

test "a window past the engine's own counts only where the rest past every fixed part holds it: no band of no cache" {
    const g: u64 = 1 << 30;
    const long: u64 = 1_048_576 + 8;
    // GLM-5.3-Flash at TP=3 under TF_KV_SHARE=32 (291 MiB fixed, 115 MiB a state), eight streams, a 1M window
    const tp3: kv.Layout = .{ .fixed_bytes = 291 * mib, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 115 * mib };
    var copies = tp3; // and without the share: GLM's copies, no slot
    copies.snap_bytes = 0;
    const own3 = tp3.pages(long) * tp3.pageBytes();
    var room = own3 + 2 * (own3 + 291 * mib); // 38.84 GiB: the room holds a third window and its fixed part
    while (room < 3 * own3 + 7 * 291 * mib) : (room += 64 * mib) { // 40.26 GiB: the rest past every fixed part does
        const p = try kv.plan(tp3, .{ .room = room, .engine = 0, .lanes = 8, .fixed = true, .cache = 16 * g, .snaps = 32, .window = long, .engine_paged = true, .slots_only = true });
        try expectEqual(@as(usize, 32), p.snaps);
        try expect(p.pages >= 2 * 2049);
        const c = try kv.plan(copies, .{ .room = room, .engine = 0, .lanes = 8, .fixed = true, .cache = 16 * g, .window = long, .engine_paged = true });
        try expectEqual(room - 2 * own3 - 7 * 291 * mib, c.copied); // two windows, then the cache gets the rest
        try expect(kv.holds(copies, .{ .room = room, .engine = 0, .lanes = 8, .fixed = true, .cache = c.copied, .window = long, .engine_paged = true }));
    }
    // TP=2 (340 MiB fixed, 180 MiB a state), four streams, a 630,000-token window: 15.66 to 16.32 GiB the same way
    const tp2: kv.Layout = .{ .fixed_bytes = 340 * mib, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 180 * mib };
    const short: u64 = 630_000 + 8;
    const own2 = tp2.pages(short) * tp2.pageBytes();
    room = own2 + (own2 + 340 * mib);
    while (room < 2 * own2 + 3 * 340 * mib) : (room += 64 * mib) {
        const p = try kv.plan(tp2, .{ .room = room, .engine = 0, .lanes = 4, .fixed = true, .cache = 16 * g, .snaps = 16, .window = short, .engine_paged = true, .slots_only = true });
        try expectEqual(@as(usize, 16), p.snaps);
    }
}

const Pin = struct { room: u64, plan: kv.Plan };

fn pinned(l: kv.Layout, b: kv.Budget, rows: []const Pin) !void {
    for (rows) |r| {
        var at = b;
        at.room = r.room;
        try expectEqual(r.plan, try kv.plan(l, at));
    }
}

test "outside that band the room's count of windows stands: GLM at TP=3 and TP=2, and Gemma 4's layout" {
    const g: u64 = 1 << 30;
    // plans where every window the room counts is held whole: below the band and past it, the band's test's layouts
    const tp3: kv.Layout = .{ .fixed_bytes = 291 * mib, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 115 * mib };
    const share3: kv.Budget = .{ .room = 0, .engine = 0, .lanes = 8, .fixed = true, .cache = 16 * g, .snaps = 32, .window = 1_048_576 + 8, .engine_paged = true, .slots_only = true };
    try pinned(tp3, share3, &.{
        .{ .room = 37 * g, .plan = .{ .lanes = 8, .snaps = 32, .copied = 0, .pages = 5046, .kept = 6339952640, .taken = 6981156864 } },
        .{ .room = 38 * g, .plan = .{ .lanes = 8, .snaps = 32, .copied = 0, .pages = 5207, .kept = 7413694464, .taken = 5907415040 } },
        .{ .room = 38 * g + 3 * g / 4, .plan = .{ .lanes = 8, .snaps = 32, .copied = 0, .pages = 5327, .kept = 8219000832, .taken = 5102108672 } },
        .{ .room = 44 * g, .plan = .{ .lanes = 8, .snaps = 32, .copied = 0, .pages = 6170, .kept = 159252480, .taken = 13161857024 } },
    });
    var copies3 = tp3;
    copies3.snap_bytes = 0;
    try pinned(copies3, .{ .room = 0, .engine = 0, .lanes = 8, .fixed = true, .cache = 16 * g, .window = 1_048_576 + 8, .engine_paged = true }, &.{
        .{ .room = 37 * g, .plan = .{ .lanes = 8, .snaps = 0, .copied = 10198712320, .pages = 4098, .kept = 0, .taken = 6981156864 } },
        .{ .room = 38 * g + 3 * g / 4, .plan = .{ .lanes = 8, .snaps = 0, .copied = 12077760512, .pages = 4098, .kept = 0, .taken = 5102108672 } },
        .{ .room = 40 * g + 3 * g / 8, .plan = .{ .lanes = 8, .snaps = 0, .copied = 125698048, .pages = 6147, .kept = 0, .taken = 17054171136 } },
        .{ .room = 44 * g, .plan = .{ .lanes = 8, .snaps = 0, .copied = 4018012160, .pages = 6147, .kept = 0, .taken = 13161857024 } },
    });
    const tp2: kv.Layout = .{ .fixed_bytes = 340 * mib, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 180 * mib };
    const share2: kv.Budget = .{ .room = 0, .engine = 0, .lanes = 4, .fixed = true, .cache = 16 * g, .snaps = 16, .window = 630_000 + 8, .engine_paged = true, .slots_only = true };
    try pinned(tp2, share2, &.{
        .{ .room = 15 * g, .plan = .{ .lanes = 4, .snaps = 16, .copied = 0, .pages = 1797, .kept = 3767795712, .taken = 0 } },
        .{ .room = 15 * g + 2 * g / 5, .plan = .{ .lanes = 4, .snaps = 16, .copied = 0, .pages = 1861, .kept = 4197292441, .taken = 0 } },
        .{ .room = 15 * g + 3 * g / 5, .plan = .{ .lanes = 4, .snaps = 16, .copied = 0, .pages = 1894, .kept = 4412040806, .taken = 0 } },
        .{ .room = 20 * g, .plan = .{ .lanes = 4, .snaps = 16, .copied = 0, .pages = 2600, .kept = 927727616, .taken = 8208777216 } },
    });
    var wide2 = share2; // a 1,048,576-token window: only the engine's own comes before the cache in these rooms
    wide2.window = 1_048_576 + 8;
    try pinned(tp2, wide2, &.{
        .{ .room = 16 * g + 2 * g / 5, .plan = .{ .lanes = 4, .snaps = 14, .copied = 0, .pages = 2078, .kept = 180459929, .taken = 0 } },
        .{ .room = 16 * g + 7 * g / 10, .plan = .{ .lanes = 4, .snaps = 16, .copied = 0, .pages = 2070, .kept = 125095116, .taken = 0 } },
        .{ .room = 17 * g + g / 5, .plan = .{ .lanes = 4, .snaps = 16, .copied = 0, .pages = 2151, .kept = 661966028, .taken = 0 } },
    });
    // Gemma 4's layout as gemma4_shared_host.zig pins it (7,864,384 B fixed, 512 KiB a page); its band: 40 to 109 MiB
    const gemma: kv.Layout = .{ .fixed_bytes = 7_864_384, .token_bytes = 2_048, .page_tokens = 256, .snap_bytes = 0 };
    try pinned(gemma, .{ .room = 0, .engine = 0, .lanes = 8, .cache = 16 * mib, .window = 4096, .engine_paged = true }, &.{
        .{ .room = 16 * mib, .plan = .{ .lanes = 1, .snaps = 0, .copied = 8388608, .pages = 16, .kept = 0, .taken = 0 } },
        .{ .room = 24 * mib, .plan = .{ .lanes = 2, .snaps = 0, .copied = 524224, .pages = 32, .kept = 0, .taken = 7864320 } },
        .{ .room = 32 * mib, .plan = .{ .lanes = 3, .snaps = 0, .copied = 1048448, .pages = 32, .kept = 0, .taken = 7340032 } },
        .{ .room = 112 * mib, .plan = .{ .lanes = 8, .snaps = 0, .copied = 3669568, .pages = 112, .kept = 0, .taken = 13107648 } },
        .{ .room = 128 * mib, .plan = .{ .lanes = 8, .snaps = 0, .copied = 12058176, .pages = 128, .kept = 0, .taken = 4719040 } },
        .{ .room = 160 * mib, .plan = .{ .lanes = 8, .snaps = 0, .copied = 16777216, .pages = 128, .kept = 0, .taken = 0 } },
    });
}

test "a --prompt-cache-gib is named for n slots only where asking it keeps them all, else the most the room holds" {
    const g: u64 = 1 << 30;
    const tp2: kv.Layout = .{ .fixed_bytes = 340 * mib, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 180 * mib };
    // TP=2, four streams, a 1M window, a 16.5 GiB room: only the engine's own window comes first, asked or not
    const b: kv.Budget = .{ .room = 16 * g + g / 2, .engine = 0, .lanes = 4, .fixed = true, .cache = 16 * g, .snaps = 16, .window = 1_048_576 + 8, .engine_paged = true, .slots_only = true };
    try expectEqual(@as(usize, 15), (try kv.plan(tp2, b)).snaps);
    const h = kv.holding(tp2, b, 16);
    try expectEqual(@as(?u64, null), h.gib); // 16 slots round up to 3 GiB, and 3 keeps the same 15
    try expectEqual(@as(usize, 15), h.most);
    var roomy = b;
    roomy.room = 24 * g;
    try expectEqual(@as(?u64, 3), kv.holding(tp2, roomy, 16).gib);
    var strip = b; // 16.69 GiB: the rest past the engine's window holds the 16 slots but not all of 3 GiB
    strip.room = 16 * g + 11 * g / 16;
    var three = strip;
    three.asked = true;
    three.cache = 3 * g;
    try expect(!kv.holds(tp2, three));
    try expectEqual(@as(usize, 16), (try kv.plan(tp2, three)).snaps); // clipped to the rest, 3 asked keeps all 16
    try expectEqual(@as(?u64, 3), kv.holding(tp2, strip, 16).gib);
    // TP=3, eight streams, just past where the rest holds a third window: 4 GiB asked holds all 32
    const tp3: kv.Layout = .{ .fixed_bytes = 291 * mib, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 115 * mib };
    const edge: kv.Budget = .{ .room = 40 * g + 5 * g / 16, .engine = 0, .lanes = 8, .fixed = true, .cache = 16 * g, .snaps = 32, .window = 1_048_576 + 8, .engine_paged = true, .slots_only = true };
    const e = kv.holding(tp3, edge, 32);
    try expectEqual(@as(?u64, 4), e.gib);
    try expectEqual(@as(usize, 32), e.most);
}

// GLM-5.3-Flash at TP=3 today, as its startup line rounds it: 327 MiB fixed, a 168 MiB slot, a 6.38 MiB page
const now3: kv.Layout = .{ .fixed_bytes = 327 * mib, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 168 * mib };
const share8: kv.Budget = .{ .room = 0, .engine = 0, .lanes = 8, .fixed = true, .cache = 16 << 30, .snaps = 32, .window = 1_048_576 + 8, .engine_paged = true, .slots_only = true };

test "a named TF_KV_SHARE's slots come where an asked cache goes, before the windows past the engine's own (#2987)" {
    const g: u64 = 1 << 30;
    // Jammed XSFP4's 42.0 GiB at TP=3, --parallel 8, TF_KV_SHARE=32: all 32 and 5,538 pages, not 8 after a third window
    try pinned(now3, share8, &.{.{ .room = 41 * g + 992 * mib, .plan = .{ .lanes = 8, .snaps = 32, .copied = 0, .pages = 5538, .kept = 9632481280, .taken = 1910243328 } }});
    var asked = share8; // the pages --prompt-cache-gib 6 leaves beside the same 32
    asked.room = 41 * g + 992 * mib;
    asked.asked = true;
    asked.cache = 6 * g;
    try expectEqual(@as(usize, 5538), (try kv.plan(now3, asked)).pages);
    // #2568's TP=2 receipt (340 MiB fixed, 180 MiB a slot, --parallel 4, a 630,000-token window): all 16, not 4
    const tp2: kv.Layout = .{ .fixed_bytes = 340 * mib, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 180 * mib };
    const share4: kv.Budget = .{ .room = 0, .engine = 0, .lanes = 4, .fixed = true, .cache = 16 * g, .snaps = 16, .window = 630_000 + 8, .engine_paged = true, .slots_only = true };
    try pinned(tp2, share4, &.{
        .{ .room = 16 * g + 2 * g / 5, .plan = .{ .lanes = 4, .snaps = 16, .copied = 0, .pages = 2022, .kept = 5271034265, .taken = 0 } },
        .{ .room = 17 * g + g / 20, .plan = .{ .lanes = 4, .snaps = 16, .copied = 0, .pages = 2126, .kept = 5968966451, .taken = 0 } },
        .{ .room = 17 * g + g / 10, .plan = .{ .lanes = 4, .snaps = 16, .copied = 0, .pages = 2134, .kept = 6022653542, .taken = 0 } },
        .{ .room = 17 * g + g / 5, .plan = .{ .lanes = 4, .snaps = 16, .copied = 0, .pages = 2151, .kept = 6130027724, .taken = 0 } },
    });
    // #2568's TP=3 layout (291 MiB fixed, 115 MiB a state) just past the third window's edge: all 32, not 0, 1 or 7
    const tp3: kv.Layout = .{ .fixed_bytes = 291 * mib, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 115 * mib };
    try pinned(tp3, share8, &.{
        .{ .room = 40 * g + 5 * g / 16, .plan = .{ .lanes = 8, .snaps = 32, .copied = 0, .pages = 5578, .kept = 9896722432, .taken = 3424387072 } },
        .{ .room = 40 * g + 3 * g / 8, .plan = .{ .lanes = 8, .snaps = 32, .copied = 0, .pages = 5588, .kept = 9963831296, .taken = 3357278208 } },
        .{ .room = 41 * g + g / 8, .plan = .{ .lanes = 8, .snaps = 32, .copied = 0, .pages = 5709, .kept = 10769137664, .taken = 2551971840 } },
    });
}

test "plans stand where today's keeps all n, without TF_KV_SHARE, beside --prompt-cache-gib and at kept_ppm" {
    const g: u64 = 1 << 30;
    // XSFP4's 37.2 GiB at TP=3: 4,780 pages and all 32 after two windows
    try pinned(now3, share8, &.{.{ .room = 37 * g + 252 * mib, .plan = .{ .lanes = 8, .snaps = 32, .copied = 0, .pages = 4780, .kept = 4561567744, .taken = 6981156864 } }});
    var asked = share8; // --prompt-cache-gib 6 beside TF_KV_SHARE=32 at Jammed's room
    asked.asked = true;
    asked.cache = 6 * g;
    try pinned(now3, asked, &.{.{ .room = 41 * g + 992 * mib, .plan = .{ .lanes = 8, .snaps = 32, .copied = 0, .pages = 5538, .kept = 805306368, .taken = 0 } }});
    var copies = now3; // TF_KV_SHARE unset (or empty: kv_prompts.sharing): GLM's copies, no slot
    copies.snap_bytes = 0;
    try pinned(copies, .{ .room = 0, .engine = 0, .lanes = 8, .fixed = true, .cache = 16 * g, .window = 1_048_576 + 8, .engine_paged = true }, &.{
        .{ .room = 37 * g + 252 * mib, .plan = .{ .lanes = 8, .snaps = 0, .copied = 10198712320, .pages = 4098, .kept = 0, .taken = 6981156864 } },
        .{ .room = 41 * g + 992 * mib, .plan = .{ .lanes = 8, .snaps = 0, .copied = 1572732928, .pages = 6147, .kept = 0, .taken = 15607136256 } },
    });
    // a family whose kept states are pool pages (kept_ppm, no slot) under TF_KV_SHARE: Gemma 4's layout as above
    const gemma: kv.Layout = .{ .fixed_bytes = 7_864_384, .token_bytes = 2_048, .page_tokens = 256, .snap_bytes = 0 };
    try pinned(gemma, .{ .room = 0, .engine = 0, .lanes = 8, .cache = 16 * mib, .window = 4096, .engine_paged = true, .kept_ppm = 1_000_000 }, &.{
        .{ .room = 24 * mib, .plan = .{ .lanes = 2, .snaps = 0, .copied = 0, .pages = 32, .kept = 524224, .taken = 7864320 } },
        .{ .room = 64 * mib, .plan = .{ .lanes = 7, .snaps = 0, .copied = 0, .pages = 37, .kept = 3145344, .taken = 5242880 } },
        .{ .room = 112 * mib, .plan = .{ .lanes = 8, .snaps = 0, .copied = 0, .pages = 118, .kept = 3669568, .taken = 13107648 } },
        .{ .room = 160 * mib, .plan = .{ .lanes = 8, .snaps = 0, .copied = 0, .pages = 160, .kept = 16777216, .taken = 0 } },
    });
}

test "with --parallel named, TF_KV_SHARE's kept states never fall as the room grows: no band, no window's edge" {
    const g: u64 = 1 << 30;
    const tp2: kv.Layout = .{ .fixed_bytes = 340 * mib, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 180 * mib };
    const Case = struct { l: kv.Layout, lanes: u32, snaps: usize, window: u64 };
    const cases = [_]Case{
        .{ .l = .{ .fixed_bytes = 291 * mib, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 115 * mib }, .lanes = 8, .snaps = 32, .window = 1_048_576 + 8 },
        .{ .l = now3, .lanes = 8, .snaps = 32, .window = 1_048_576 + 8 },
        .{ .l = tp2, .lanes = 4, .snaps = 16, .window = 630_000 + 8 },
        .{ .l = tp2, .lanes = 4, .snaps = 16, .window = 1_048_576 + 8 },
    };
    for (cases) |c| {
        var last: usize = 0;
        var room: u64 = 8 * g;
        while (room <= 64 * g) : (room += 64 * mib) {
            const p = kv.plan(c.l, .{ .room = room, .engine = 0, .lanes = c.lanes, .fixed = true, .cache = 16 * g, .snaps = c.snaps, .window = c.window, .engine_paged = true, .slots_only = true }) catch continue; // the streams don't fit yet
            try expect(p.snaps >= last);
            last = p.snaps;
        }
        try expectEqual(c.snaps, last);
    }
}
