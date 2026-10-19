const cuda = @import("cuda");
const tp = @import("tp");
export fn compileTp(d: *const cuda.Driver, cfg: *const tp.Config, g: *cuda.tp_group.Group, stream: *const cuda.Stream) void {
    var session: tp.Session = undefined;
    session.open(d, cfg.*, &.{}) catch return;
    defer session.close();
    var old = tp.existing.Existing.init(g, d, cfg.*) catch return;
    defer old.deinit();
    old.agree(d, stream.*, @splat(0)) catch {};
    old.comm().check() catch {};
}
