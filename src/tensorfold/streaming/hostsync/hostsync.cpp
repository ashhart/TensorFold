// The GPU signals and waits for the host inside MLX's command stream through an MTLSharedEvent, never an MLX eval.

#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "mlx/backend/metal/device.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"

namespace nb = nanobind;
namespace mx = mlx::core;

// One monotonic event shared by the GPU and the host: each side signals increasing values.
struct Channel {
  NS::SharedPtr<MTL::SharedEvent> event;
  Channel() : event(NS::TransferPtr(mx::metal::device(mx::Device::gpu).mtl_device()->newSharedEvent())) {}
  uint64_t value() const { return event->signaledValue(); }
  void signal(uint64_t v) { event->setSignaledValue(v); }
  bool wait(uint64_t v, uint64_t timeout_ms) {
    nb::gil_scoped_release release;
    return event->waitUntilSignaledValue(v, timeout_ms);
  }
};

// Passes its input through; on the GPU it first ends the compute encoder, then waits for or signals the channel.
class Mark : public mx::UnaryPrimitive {
 public:
  Mark(mx::Stream s, std::shared_ptr<Channel> channel, uint64_t value, bool wait)
      : mx::UnaryPrimitive(s), channel_(std::move(channel)), value_(value), wait_(wait) {}
  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("hostsync marks run on the GPU stream only");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    out.copy_shared_buffer(inputs[0]);
    auto& enc = mx::metal::get_command_encoder(stream());
    enc.end_encoding();
    if (wait_) {
      enc.get_command_buffer()->encodeWait(channel_->event.get(), value_);
    } else {
      enc.get_command_buffer()->encodeSignalEvent(channel_->event.get(), value_);
    }
  }
  const char* name() const override { return wait_ ? "HostWait" : "HostSignal"; }
  bool is_equivalent(const mx::Primitive& other) const override {
    auto* o = dynamic_cast<const Mark*>(&other);
    return o && o->channel_ == channel_ && o->value_ == value_ && o->wait_ == wait_;
  }

 private:
  std::shared_ptr<Channel> channel_;
  uint64_t value_;
  bool wait_;
};

mx::array mark(const mx::array& x, std::shared_ptr<Channel> channel, uint64_t value, bool wait) {
  auto s = mx::default_stream(mx::Device::gpu);
  return mx::array(x.shape(), x.dtype(), std::make_shared<Mark>(s, std::move(channel), value, wait), {x});
}

// Reads nbytes of fd at file_offset into an evaluated array's buffer at byte_offset (the caller owns the timing).
void pread_into(mx::array& a, size_t byte_offset, int fd, int64_t file_offset, size_t nbytes) {
  if (!a.is_available() || byte_offset + nbytes > a.nbytes()) {
    throw std::out_of_range("pread_into: the array is not evaluated or the range passes its end");
  }
  auto* dst = a.data<uint8_t>() + byte_offset;
  nb::gil_scoped_release release;
  size_t done = 0;
  while (done < nbytes) {
    ssize_t got = pread(fd, dst + done, nbytes - done, file_offset + static_cast<int64_t>(done));
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0) throw std::runtime_error(std::string("pread_into: short read: ") + std::strerror(errno));
    done += static_cast<size_t>(got);
  }
}

// Copies bytes into an evaluated array's buffer at byte_offset (host writes the GPU reads after a later wait).
void write_into(mx::array& a, size_t byte_offset, nb::bytes data) {
  if (!a.is_available() || byte_offset + data.size() > a.nbytes()) {
    throw std::out_of_range("write_into: the array is not evaluated or the range passes its end");
  }
  std::memcpy(a.data<uint8_t>() + byte_offset, data.c_str(), data.size());
}

// Reads nbytes of an evaluated array's buffer at byte_offset without waiting on MLX (after a GPU signal).
nb::bytes peek(mx::array& a, size_t byte_offset, size_t nbytes) {
  if (!a.is_available() || byte_offset + nbytes > a.nbytes()) {
    throw std::out_of_range("peek: the array is not evaluated or the range passes its end");
  }
  return nb::bytes(reinterpret_cast<const char*>(a.data<uint8_t>() + byte_offset), nbytes);
}

NB_MODULE(_hostsync, m) {
  nb::class_<Channel>(m, "Channel")
      .def(nb::init<>())
      .def("value", &Channel::value)
      .def("signal", &Channel::signal)
      .def("wait", &Channel::wait, nb::arg("value"), nb::arg("timeout_ms"));
  m.def("gpu_signal", [](const mx::array& x, std::shared_ptr<Channel> c, uint64_t v) { return mark(x, c, v, false); });
  m.def("gpu_wait", [](const mx::array& x, std::shared_ptr<Channel> c, uint64_t v) { return mark(x, c, v, true); });
  m.def("pread_into", &pread_into);
  m.def("write_into", &write_into);
  m.def("peek", &peek);
}
