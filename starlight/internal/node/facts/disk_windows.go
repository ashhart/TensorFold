package facts

import "golang.org/x/sys/windows"

func diskFree(dir string) uint64 {
	p, err := windows.UTF16PtrFromString(dir)
	if err != nil {
		return 0
	}
	var avail, total, free uint64
	if windows.GetDiskFreeSpaceEx(p, &avail, &total, &free) != nil {
		return 0
	}
	return avail
}
