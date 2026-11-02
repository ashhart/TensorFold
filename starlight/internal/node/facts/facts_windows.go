package facts

import (
	"context"
	"fmt"
	"unsafe"

	"golang.org/x/sys/windows"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
)

type memoryStatusEx struct {
	Length               uint32
	MemoryLoad           uint32
	TotalPhys            uint64
	AvailPhys            uint64
	TotalPageFile        uint64
	AvailPageFile        uint64
	TotalVirtual         uint64
	AvailVirtual         uint64
	AvailExtendedVirtual uint64
}

var globalMemoryStatusEx = windows.NewLazySystemDLL("kernel32.dll").NewProc("GlobalMemoryStatusEx")

func memoryBytes(context.Context) uint64 {
	var m memoryStatusEx
	m.Length = uint32(unsafe.Sizeof(m))
	if r, _, _ := globalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&m))); r == 0 {
		return 0
	}
	return m.TotalPhys
}

func osVersion(context.Context) string {
	v := windows.RtlGetVersion()
	return fmt.Sprintf("Windows %d.%d.%d", v.MajorVersion, v.MinorVersion, v.BuildNumber)
}

func gpus(ctx context.Context, _ uint64) ([]wire.GPU, error) {
	return nvidiaGPUs(ctx)
}
