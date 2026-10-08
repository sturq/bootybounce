#!/usr/bin/env python3
"""Turn a video of a person into a pet folder for bootybounce.

    python3 make_pet.py VIDEO NAME [--caption Y0,Y1] [--box X0,Y0,X1,Y1] [--loop A,B]

Needs numpy, opencv-python-headless and imageio-ffmpeg, plus a ComfyUI with the
ComfyUI-RMBG nodes (BEN2) for the cutout. Writes pets/NAME/ (frames, hit.json,
pet.cfg) and a labelled preview video into the work folder.

--caption  rows of a burned-in caption to paint out before cutting (static white text)
--box      crop around the person in source pixels; default: found from a first pass
--loop     ping-pong between these two source frames; default: chosen automatically
"""
import argparse, json, os, subprocess, sys, time, urllib.parse, urllib.request, uuid
import cv2
import imageio_ffmpeg
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('video')
p.add_argument('name')
p.add_argument('--caption')
p.add_argument('--box')
p.add_argument('--loop')
p.add_argument('--model', default='BEN2')
p.add_argument('--comfy', default='http://127.0.0.1:8188')
p.add_argument('--work', default=None)
p.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pets'))
args = p.parse_args()
W = args.work or os.path.join(os.path.dirname(os.path.abspath(args.video)), args.name + '-work')
FF = imageio_ffmpeg.get_ffmpeg_exe()
ints = lambda s: [int(v) for v in s.split(',')]


def comfy(path, data=None, ctype='application/json'):
    r = urllib.request.Request(args.comfy + path, data=data, headers={'Content-Type': ctype})
    with urllib.request.urlopen(r, timeout=120) as f:
        return f.read()


def cut_out(frames, box, out):
    """Person mask for each (name, image) through ComfyUI, cropped to box. Cached in out/."""
    os.makedirs(out, exist_ok=True)
    x0, y0, x1, y1 = box
    opts = {'sensitivity': 1.0, 'process_res': 1024, 'mask_blur': 0, 'mask_offset': 0, 'invert_output': False,
            'refine_foreground': False, 'background': 'Alpha', 'background_color': '#222222'}
    jobs = {}
    for n, img in frames:
        if os.path.exists(f'{out}/{n}'):
            continue
        b = uuid.uuid4().hex
        body = (f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="bb_{n}"\r\n'
                f'Content-Type: image/png\r\n\r\n').encode() + cv2.imencode('.png', img[y0:y1, x0:x1])[1].tobytes() + \
            f'\r\n--{b}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n--{b}--\r\n'.encode()
        up = json.loads(comfy('/upload/image', body, f'multipart/form-data; boundary={b}'))['name']
        g = {'1': {'class_type': 'LoadImage', 'inputs': {'image': up}},
             '2': {'class_type': 'RMBG', 'inputs': {'image': ['1', 0], 'model': args.model, **opts}},
             '3': {'class_type': 'MaskToImage', 'inputs': {'mask': ['2', 1]}},
             '4': {'class_type': 'SaveImage', 'inputs': {'images': ['3', 0], 'filename_prefix': 'bootybounce/' + args.name}}}
        jobs[json.loads(comfy('/prompt', json.dumps({'prompt': g}).encode()))['prompt_id']] = n
    while jobs:
        time.sleep(2)
        for pid, n in list(jobs.items()):
            h = json.loads(comfy('/history/' + pid)).get(pid)
            if not h:
                continue
            if h.get('status', {}).get('status_str') == 'error':
                sys.exit('ComfyUI failed on %s: %s' % (n, json.dumps(h['status'])[:800]))
            imgs = [i for o in h.get('outputs', {}).values() for i in o.get('images', [])]
            if imgs:
                q = urllib.parse.urlencode({k: imgs[0].get(k, '') for k in ('filename', 'subfolder', 'type')})
                open(f'{out}/{n}', 'wb').write(comfy('/view?' + q))
                del jobs[pid]
    return [cv2.imread(f'{out}/{n}', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255 for n, _ in frames]


def blur_fusion(I, al, r):
    """Foreground colour estimate (Germer et al. 2020): takes the old background out of soft edges."""
    def step(F, B, r):
        ba = cv2.blur(al, (r, r))[..., None]
        bF = cv2.blur(F * al[..., None], (r, r)) / (ba + 1e-5)
        bB = cv2.blur(B * (1 - al[..., None]), (r, r)) / (1 - ba + 1e-5)
        A = al[..., None]
        return np.clip(bF + A * (I - A * bF - (1 - A) * bB), 0, 1), bB
    F, B = step(I, I, r)
    return step(F, B, 6)[0]


# 1. frames
os.makedirs(f'{W}/src', exist_ok=True)
if not os.listdir(f'{W}/src'):
    subprocess.run([FF, '-v', 'error', '-i', args.video, '-fps_mode', 'passthrough', f'{W}/src/%04d.png'], check=True)
names = sorted(os.listdir(f'{W}/src'))
src = [cv2.imread(f'{W}/src/{n}') for n in names]
fps = float(subprocess.run([FF, '-i', args.video], capture_output=True, text=True).stderr.split(' fps')[0].split()[-1])
N = len(src)
print(N, 'frames', src[0].shape[1], 'x', src[0].shape[0], '%.3f fps' % fps)

# 2. caption: it stays white in every frame, so even a hand passing behind it never darkens it
if args.caption:
    y0, y1 = ints(args.caption)
    mn = np.min(np.stack([f.min(axis=2) for f in src]), axis=0)
    mask = np.zeros_like(mn)
    mask[y0:y1] = (mn[y0:y1] >= 215) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    src = [cv2.inpaint(f, mask, 4, cv2.INPAINT_TELEA) for f in src]

# 3. cutout: a rough pass on every 10th full frame finds the box, then the real pass on the crop
H, Wd = src[0].shape[:2]
if args.box:
    box = ints(args.box)
else:
    rough = cut_out([(n, f) for n, f in zip(names, src)][::10], (0, 0, Wd, H), f'{W}/mask-rough')
    ys, xs = np.nonzero(np.max(rough, axis=0) > 0.3)
    m = int(0.08 * (ys.max() - ys.min()))
    box = (max(xs.min() - m, 0), max(ys.min() - m, 0), min(xs.max() + m, Wd), min(ys.max() + m, H))
print('box', ','.join(map(str, box)))
alpha = cut_out(list(zip(names, src)), box, f'{W}/mask-{args.model}-' + '_'.join(map(str, box)))
x0, y0, x1, y1 = box
crops = [f[y0:y1, x0:x1].astype(np.float32) / 255 for f in src]

# 4. loop: a natural cut if one is close enough, otherwise ping-pong between two still poses
feat = np.stack([np.concatenate([cv2.resize(c * a[..., None] + 0.5 * (1 - a[..., None]), (64, 96), interpolation=cv2.INTER_AREA).ravel(),
                                 2 * cv2.resize(a, (64, 96), interpolation=cv2.INTER_AREA).ravel()]) for c, a in zip(crops, alpha)])
sq = (feat ** 2).sum(1)
D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * feat @ feat.T, 0) / feat.shape[1])
step = np.median(np.diag(D, 1))
pingpong = True
if args.loop:
    a, b = ints(args.loop)
else:
    cuts = [(sum(D[i + k, j + k] for k in (-1, 0, 1)) / 3, i, j)
            for i in range(1, N // 2) for j in range(i + int(1.5 * fps), N - 1)]
    seam, a, b = min(cuts)
    print('best cut %d..%d seam %.4f, normal step %.4f' % (a, b, seam, step))
    if seam <= 1.5 * step:
        pingpong = False
        b -= 1  # frame b is where frame a would come next
    else:
        motion = lambda k: D[k - 1, k] + D[k, k + 1]
        a = min(range(1, N // 3), key=motion)
        b = min(range(N // 2, N - 1), key=motion)
print('loop', 'ping-pong' if pingpong else 'cut', a, b, '%d frames' % (b - a + 1))

# 5. frames, click polygons, preview
rgba = []
for k in range(a, b + 1):
    al = np.clip((alpha[k] - 0.04) / 0.92, 0, 1)  # drops the faint haze around the body
    _, lab, st, _ = cv2.connectedComponentsWithStats((al > 0.02).astype(np.uint8))
    body = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)  # stray blobs (lights, old caption) go
    al *= cv2.dilate(body, np.ones((7, 7), np.uint8))
    rgba.append(np.dstack([blur_fusion(crops[k], al, 61), al]))
ys, xs = np.nonzero(np.max([f[..., 3] for f in rgba], axis=0) > 0.02)
m = 6
cy0, cy1, cx0, cx1 = max(ys.min() - m, 0), ys.max() + m + 1, max(xs.min() - m, 0), xs.max() + m + 1
out = os.path.join(args.out, args.name)
os.makedirs(f'{out}/frames', exist_ok=True)
for old in os.listdir(f'{out}/frames'):
    os.remove(f'{out}/frames/{old}')
hit = []
n = len(rgba)
for i, f in enumerate(rgba):
    f = f[cy0:cy1, cx0:cx1]
    cv2.imwrite(f'{out}/frames/%04d.png' % (i + 1), (f * 255 + 0.5).astype(np.uint8))
    # the click/visible region also covers the neighbour frames, so a frame shown a tick late is never clipped
    nb = [rgba[j][cy0:cy1, cx0:cx1, 3] for j in range(max(i - 2, 0), min(i + 3, n))]
    if not pingpong and (i < 2 or i > n - 3):
        nb += [rgba[j % n][cy0:cy1, cx0:cx1, 3] for j in range(i - 2, i + 3)]
    u = (np.max(nb, axis=0) > 0.02).astype(np.uint8) * 255
    u = cv2.morphologyEx(cv2.dilate(u, np.ones((5, 5), np.uint8)), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    c = max(cv2.findContours(u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    hit.append([int(v) for v in cv2.approxPolyDP(c, 1.5, True).ravel()])
json.dump(hit, open(f'{out}/hit.json', 'w'), separators=(',', ':'))
open(f'{out}/pet.cfg', 'w').write('[pet]\nfps=%.3f\npingpong=%s\n' % (fps, 'true' if pingpong else 'false'))
order = list(range(n)) + (list(range(n - 2, 0, -1)) if pingpong else [])
small = [cv2.resize(f, (64, 96), interpolation=cv2.INTER_AREA) for f in rgba]
steps = [np.sqrt(((small[order[t]] - small[order[(t + 1) % len(order)]]) ** 2).mean()) for t in range(len(order))]
print('%s: %d frames %dx%d, loop %.2f s, step median %.4f max %.4f' % (
    out, n, cx1 - cx0, cy1 - cy0, len(order) / fps, np.median(steps), max(steps)))

pv = f'{W}/preview'
os.makedirs(pv, exist_ok=True)
for t, i in enumerate(order * 2):
    f = rgba[i][cy0:cy1, cx0:cx1]
    A = f[..., 3:]
    tiles = []
    for bg, label in (((0.14, 0.12, 0.12), 'dark'), ((0.9, 0.93, 0.93), 'light')):
        tile = ((f[..., :3] * A + np.array(bg) * (1 - A)) * 255).astype(np.uint8)
        cv2.putText(tile, '%s %s %s' % (args.name, 'ping-pong' if pingpong else 'cut', label), (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 230), 2)
        tiles.append(tile)
    cv2.imwrite(f'{pv}/%04d.png' % t, np.hstack(tiles))
subprocess.run([FF, '-v', 'error', '-y', '-framerate', '%.5f' % fps, '-i', f'{pv}/%04d.png', '-vf',
                'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p',
                f'{W}/{args.name}_preview_2loops.mp4'], check=True)
print('preview', f'{W}/{args.name}_preview_2loops.mp4')
