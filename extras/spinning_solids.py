"""Spinning ASCII Platonic solids. Run: uv run python extras/spinning_solids.py

Controls (interactive TTY):
    <- / ->   cycle solids, in order of increasing vertex count
    up / down  grow / shrink the solid
    q          quit  (Ctrl-C also works)

The canvas auto-fits the terminal every frame, so resizing the window just works.
Each solid is tagged with the binary polyhedral group it belongs to (2T/2O/2I),
tying the animation back to the thesis atlas.
"""
import math, os, shutil, sys, time

phi = (1 + 5**0.5) / 2
inv = 1 / phi


def _ico():
    V = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            V += [(0, s1, s2*phi), (s1, s2*phi, 0), (s2*phi, 0, s1)]
    return V


def _dodeca():
    V = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            V += [(0, s1*inv, s2*phi), (s1*inv, s2*phi, 0), (s2*phi, 0, s1*inv)]
    return V


def normalize(V):
    """Scale a solid so its circumradius is 1, so every solid looks the same size."""
    r = math.sqrt(sum(c*c for c in V[0]))
    return [tuple(c/r for c in v) for v in V]


def make_edges(V):
    """Edges of a Platonic solid are exactly its shortest vertex-vertex pairs."""
    n = len(V)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    d2 = {p: sum((V[p[0]][k]-V[p[1]][k])**2 for k in range(3)) for p in pairs}
    m = min(d2.values())
    return [p for p in pairs if d2[p] <= m * 1.0001 + 1e-9]


# (name, group, vertices) ordered by increasing vertex count.
_RAW = [
    ("Tetrahedron", "2T", [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]),
    ("Octahedron",  "2O", [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]),
    ("Cube",        "2O", [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]),
    ("Icosahedron", "2I", _ico()),
    ("Dodecahedron", "2I", _dodeca()),
]
SOLIDS = [(name, grp, (lambda v: (v, make_edges(v)))(normalize(verts)))
          for name, grp, verts in _RAW]

SHADES = ' .·-:=+*o#%@'


def rotate(p, ax, ay, az):
    x, y, z = p
    y, z = y*math.cos(ax) - z*math.sin(ax), y*math.sin(ax) + z*math.cos(ax)
    x, z = x*math.cos(ay) + z*math.sin(ay), -x*math.sin(ay) + z*math.cos(ay)
    x, y = x*math.cos(az) - y*math.sin(az), x*math.sin(az) + y*math.cos(az)
    return (x, y, z)


def project(p, W, H, scale):
    d = 6
    f = scale / (d - p[2])
    return (W/2 + p[0]*f*2, H/2 - p[1]*f)


def frame(V, E, t, W, H, scale):
    grid = [[' ']*W for _ in range(H)]
    R = [rotate(v, t*0.61, t*0.83, t*0.27) for v in V]
    P = [project(p, W, H, scale) for p in R]
    for i, j in E:
        x0, y0 = P[i]; x1, y1 = P[j]
        z = (R[i][2] + R[j][2]) / 2
        ch = SHADES[max(0, min(len(SHADES)-1, int((z + 1) / 2 * (len(SHADES)-1))))]
        steps = int(max(abs(x1-x0), abs(y1-y0))) + 1
        for s in range(steps + 1):
            u = s / steps
            x, y = int(x0 + (x1-x0)*u), int(y0 + (y1-y0)*u)
            if 0 <= x < W and 0 <= y < H:
                grid[y][x] = ch
    for i, (x, y) in enumerate(P):
        xi, yi = int(x), int(y)
        if 0 <= xi < W and 0 <= yi < H and R[i][2] > -0.4:
            grid[yi][xi] = '@'
    return '\n'.join(''.join(r) for r in grid)


def base_fit(W, H):
    """Default scale that fits a unit-circumradius solid into the canvas."""
    return min(1.3 * W, 2.6 * H)


def _poll_keys(fd):
    """Drain stdin (non-blocking) and decode arrow keys + single chars."""
    try:
        data = os.read(fd, 64).decode('latin-1')
    except (BlockingIOError, OSError):
        return []
    out, i = [], 0
    arrows = {'A': 'up', 'B': 'down', 'C': 'right', 'D': 'left'}
    while i < len(data):
        if data[i] == '\033' and data[i+1:i+2] == '[':
            out.append(arrows.get(data[i+2:i+3], ''))
            i += 3
        else:
            out.append(data[i])
            i += 1
    return out


def run_interactive():
    import termios, tty, fcntl
    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    old_fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    tty.setcbreak(fd)
    fcntl.fcntl(fd, fcntl.F_SETFL, old_fl | os.O_NONBLOCK)
    sys.stdout.write('\033[2J\033[?25l')

    idx, zoom, t, prev_dim = 3, 0.9, 0.0, None  # start on the icosahedron
    try:
        while True:
            for k in _poll_keys(fd):
                if k in ('q', '\x03', '\x04'):
                    return
                elif k == 'up':
                    zoom = min(4.0, zoom * 1.1)
                elif k == 'down':
                    zoom = max(0.2, zoom / 1.1)
                elif k == 'left':
                    idx = (idx - 1) % len(SOLIDS)
                elif k == 'right':
                    idx = (idx + 1) % len(SOLIDS)

            cols, rows = shutil.get_terminal_size((120, 40))
            W, H = max(20, cols), max(10, rows - 1)
            if (W, H) != prev_dim:
                sys.stdout.write('\033[2J')
                prev_dim = (W, H)

            name, grp, (V, E) = SOLIDS[idx]
            scale = base_fit(W, H) * zoom
            status = (f"  {name} · {len(V)} vertices · group {grp}"
                      f"   │   ←/→ solid · ↑/↓ zoom (×{zoom:4.2f}) · q quit")
            sys.stdout.write('\033[H' + frame(V, E, t, W, H, scale)
                             + '\n' + status[:W].ljust(W))
            sys.stdout.flush()
            time.sleep(0.04)
            t += 0.05
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        fcntl.fcntl(fd, fcntl.F_SETFL, old_fl)
        sys.stdout.write('\033[?25h\033[2J\033[H')
        sys.stdout.flush()


def smoke_test():
    """Non-TTY self-check: render every solid and confirm it fits the canvas."""
    W, H = 120, 40
    ok = True
    for name, grp, (V, E) in SOLIDS:
        scale = base_fit(W, H) * 0.9
        worst = 0.0
        for tt in (0.0, 0.5, 1.0, 1.7, 2.3):
            R = [rotate(v, tt*0.61, tt*0.83, tt*0.27) for v in V]
            P = [project(p, W, H, scale) for p in R]
            for x, y in P:
                worst = max(worst, abs(x - W/2) / (W/2), abs(y - H/2) / (H/2))
        s = frame(V, E, 1.0, W, H, scale)
        rows = len(s.splitlines())
        fits = worst <= 1.0 and rows == H
        ok = ok and fits
        print(f"{name:<13} {grp}  {len(V):>2} verts  {len(E):>2} edges  "
              f"fill={worst:4.2f}  {'ok' if fits else 'CLIP'}")
    print("all solids fit." if ok else "WARNING: clipping at default zoom.")


def main():
    if sys.stdout.isatty() and sys.stdin.isatty():
        run_interactive()
    else:
        smoke_test()


if __name__ == '__main__':
    main()
