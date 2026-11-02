package engine

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// followerGrace bounds how long a follower rank gets to exit once its group is over.
const followerGrace = 10 * time.Second

// process is one running engine and the port it announced; a lead process also owns its follower ranks.
type process struct {
	cmd       *exec.Cmd
	port      chan int
	done      chan struct{}
	err       error
	pidFile   string
	log       *rotating
	once      sync.Once
	startErr  error
	followers []*process
	mu        sync.Mutex
	cause     string
	last      []string
}

// lastLines is how many of the engine's final output lines a failure carries, so the console shows why it stopped.
const lastLines = 3

// startGroup launches follower ranks first and rank 0 last; the group ends, and is stopped whole, when any rank exits.
func startGroup(bin string, ranks []rankSpec, dir, served string, logMax int64, job *jobGroup) (*process, error) {
	var followers []*process
	for _, r := range ranks[1:] {
		f, err := launch(bin, r.args, r.env, dir, served, "engine.rank"+strconv.Itoa(r.rank), logMax, job, nil)
		if err != nil {
			for _, g := range followers {
				g.stop(followerGrace)
			}
			return nil, fmt.Errorf("rank %d: %w", r.rank, err)
		}
		followers = append(followers, f)
	}
	lead, err := launch(bin, ranks[0].args, ranks[0].env, dir, served, "engine", logMax, job, followers)
	if err != nil {
		for _, g := range followers {
			g.stop(followerGrace)
		}
		return nil, err
	}
	return lead, nil
}

// launch starts bin with args in dir, logging its output as name.log and watching for the serving line.
func launch(bin string, args, env []string, dir, served, name string, logMax int64, job *jobGroup, followers []*process) (*process, error) {
	log, err := openRotating(filepath.Join(dir, name+".log"), logMax)
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(bin, args...)
	cmd.Dir = dir
	cmd.Env = env
	cmd.Stdin = nil
	setProcAttr(cmd)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Close()
		return nil, err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		log.Close()
		return nil, err
	}
	p := &process{cmd: cmd, port: make(chan int, 1), done: make(chan struct{}), log: log, pidFile: filepath.Join(dir, name+".pid"), followers: followers}
	if err := cmd.Start(); err != nil {
		log.Close()
		return nil, fmt.Errorf("start engine: %w", err)
	}
	if err := job.add(cmd.Process); err != nil {
		p.startErr = fmt.Errorf("attach engine to job object: %w", err)
		_ = cmd.Process.Kill()
	}
	_ = os.WriteFile(p.pidFile, []byte(strconv.Itoa(cmd.Process.Pid)+"\n"+keyFileArg(args)+"\n"), 0o600)
	var wg sync.WaitGroup
	wg.Add(2)
	go p.scan(stdout, served, &wg)
	go p.scan(stderr, served, &wg)
	go func() {
		wg.Wait()
		p.err = cmd.Wait()
		for _, f := range followers {
			f.stop(followerGrace)
		}
		log.Close()
		_ = os.Remove(p.pidFile)
		close(p.done)
	}()
	for i, f := range followers {
		go p.watchFollower(i+1, f)
	}
	return p, nil
}

// watchFollower ends the group when a follower rank exits first, since rank 0 cannot serve without it.
func (p *process) watchFollower(rank int, f *process) {
	select {
	case <-f.done:
		p.mu.Lock()
		if p.cause == "" {
			p.cause = fmt.Sprintf("rank %d %s", rank, f.exitError())
		}
		p.mu.Unlock()
		terminate(p.cmd.Process)
	case <-p.done:
	}
}

func keyFileArg(args []string) string {
	for i, a := range args {
		if a == "--api-key-file" && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}

func (p *process) scan(r io.Reader, served string, wg *sync.WaitGroup) {
	defer wg.Done()
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		_, _ = p.log.Write([]byte(time.Now().UTC().Format(time.RFC3339Nano) + " " + line + "\n"))
		if t := strings.TrimSpace(line); t != "" {
			p.mu.Lock()
			p.last = append(p.last, t[:min(len(t), 300)])
			if len(p.last) > lastLines {
				p.last = p.last[1:]
			}
			p.mu.Unlock()
		}
		if name, port, ok := ParseServing(line); ok && name == served {
			p.once.Do(func() { p.port <- port })
		}
	}
	_, _ = io.Copy(io.Discard, r)
}

// stop asks the engine to exit, waits up to grace, then kills it.
func (p *process) stop(grace time.Duration) {
	select {
	case <-p.done:
		return
	default:
	}
	terminate(p.cmd.Process)
	select {
	case <-p.done:
	case <-time.After(grace):
		kill(p.cmd.Process)
		<-p.done
	}
}

// exitError describes how the process ended.
func (p *process) exitError() string {
	if p.startErr != nil {
		return p.startErr.Error()
	}
	p.mu.Lock()
	cause, last := p.cause, strings.Join(p.last, " / ")
	p.mu.Unlock()
	if cause != "" {
		return cause
	}
	msg := "engine exited"
	if p.err != nil {
		msg += ": " + p.err.Error()
	}
	if last != "" {
		msg += " (" + last + ")"
	}
	return msg
}
