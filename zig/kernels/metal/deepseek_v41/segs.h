// A slot's state at one layer of a launch over several slots, then its rows from `base` (segs.zig's Seg, tested).
struct TfDsv41Seg {
  device uchar* window;
  device uchar* pool;
  device uchar* keys;
  device float* carry;
  device float* inputs;
  device uchar* comp;
  device uchar* ikeys;
  uint base, rows, nkeys, start;
};
