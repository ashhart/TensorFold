package engine

import (
	"errors"
	"fmt"
	"net"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
)

var devicePattern = regexp.MustCompile(`^(GPU-[0-9A-Za-z-]{4,64}|[0-9]{1,2})$`)

// rankSpec is one engine process of a replica: rank 0 serves HTTP and the others follow it.
type rankSpec struct {
	rank int
	args []string
	env  []string
}

// checkDevices refuses device lists and rank groups the scheduler could not have produced for this spec.
func checkDevices(a wire.Assignment) error {
	if a.Group != nil {
		if err := a.Group.Validate(a.Spec); err != nil {
			return err
		}
	} else if a.Spec.Span() > 1 {
		return errors.New("a replica spanning nodes arrived without its rank group")
	}
	if len(a.Devices) == 0 {
		if a.Spec.GPUs > 1 {
			return errors.New("a multi-rank replica arrived without its GPUs")
		}
		return nil
	}
	if len(a.Devices) != a.Spec.Width() {
		return fmt.Errorf("replica needs %d GPUs but was given %d", a.Spec.Width(), len(a.Devices))
	}
	for i, d := range a.Devices {
		if !devicePattern.MatchString(d) || slices.Contains(a.Devices[:i], d) {
			return fmt.Errorf("device %q is not a GPU index or UUID, or is listed twice", d)
		}
	}
	return nil
}

// follower reports a member that runs ranks for a group led from another node, so nothing here serves.
func follower(a wire.Assignment) bool { return a.Group != nil && a.Group.Member > 0 }

// rankSpecs lays out this node's ranks: pinned to their GPUs, numbered from the group's first, told where rank 0 is.
func rankSpecs(a wire.Assignment, args, env []string, port int) []rankSpec {
	sc, tp := wire.RankSchemes[a.Spec.TensorParallel]
	if !tp {
		if len(a.Devices) == 0 {
			return []rankSpec{{args: args, env: env}}
		}
		return []rankSpec{{args: args, env: pin(env, a.Devices...)}}
	}
	g := wire.Group{World: a.Spec.Ranks(), Master: "127.0.0.1", Port: port}
	if a.Group != nil {
		g = *a.Group
	}
	out := make([]rankSpec, max(len(a.Devices), 1))
	for i := range out {
		rank, x, e := g.FirstRank+i, slices.Clip(args), slices.Clip(env)
		at, world, rk := net.JoinHostPort(g.Master, strconv.Itoa(g.Port)), strconv.Itoa(g.World), strconv.Itoa(rank)
		switch {
		case len(a.Devices) > 0 && sc.Meet == wire.MeetFlags:
			e, x = pin(e, a.Devices...), append(x, "--device", strconv.Itoa(i))
		case len(a.Devices) > 0:
			e = pin(e, a.Devices[i])
		}
		switch sc.Meet {
		case wire.MeetNCCL:
			e = append(e, sc.Prefix+"_RANK="+rk, sc.Prefix+"_WORLD="+world, "NCCL_COMM_ID="+at)
		case wire.MeetFabric:
			e = append(e, "TF_TP_RANK="+rk, "TF_TP_WORLD="+world, "TF_TP_MASTER="+g.Master, "TF_TP_PORT="+strconv.Itoa(g.Port), "TF_TP_DEVICE=0")
		case wire.MeetFlags:
			x = append(x, "--tp", world, "--rank", rk, "--master", g.Master, "--master-port", strconv.Itoa(g.Port))
		}
		out[i] = rankSpec{rank: rank, args: x, env: e}
	}
	return out
}

// pin limits a process to devices in both runtimes' lists, since HIP reads HIP_VISIBLE_DEVICES before CUDA's.
func pin(env []string, devices ...string) []string {
	list := strings.Join(devices, ",")
	return append(slices.Clip(env), "CUDA_VISIBLE_DEVICES="+list, "CUDA_DEVICE_ORDER=PCI_BUS_ID", "HIP_VISIBLE_DEVICES="+list)
}

// commPort finds a loopback port whose block of wire.RankPorts is free, since an engine's rendezvous uses the block.
func commPort() (int, error) {
	for i := 0; i < 32; i++ {
		a, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			return 0, err
		}
		p := a.Addr().(*net.TCPAddr).Port
		a.Close()
		if p+wire.RankPorts <= 65535 && freeBlock(p) {
			return p, nil
		}
	}
	return 0, errors.New("no free block of loopback ports for the rank link")
}

func freeBlock(p int) bool {
	for q := p; q < p+wire.RankPorts; q++ {
		l, err := net.Listen("tcp", "127.0.0.1:"+strconv.Itoa(q))
		if err != nil {
			return false
		}
		defer l.Close()
	}
	return true
}
