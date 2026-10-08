#!/usr/bin/env python3
# Step 2: the motion to copy, from make_pet.py's work folder of the source clip: frames 0..LAST at 16 fps,
# 576x768, a DWPose skeleton per frame (comfyui_controlnet_aux) and a 512x512 face crop per frame.
# Run once per person; every outfit reuses it. FETCH=1 only downloads results that ComfyUI already made.
import cv2, json, numpy as np, os, sys, time, urllib.parse, urllib.request, uuid
W = os.environ.get('WORK', 'work')  # make_pet.py work folder of the source clip
BASE = os.environ.get('COMFY', 'http://127.0.0.1:8188')
BX, BY = 100, 380                      # mask crops start here in the source frame
SRC_FPS, FPS, LAST = 30000 / 1001, 16, 186
GW, GH = 576, 768
D = f'{W}/drive'
for sub in ('rgb', 'pose', 'kp', 'face'):
    os.makedirs(f'{D}/{sub}', exist_ok=True)


def call(path, data=None, ctype='application/json'):
    r = urllib.request.Request(BASE + path, data=data, headers={'Content-Type': ctype})
    with urllib.request.urlopen(r, timeout=120) as f:
        return f.read()


def upload(name, png):
    b = uuid.uuid4().hex
    body = (f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\nContent-Type: image/png\r\n\r\n').encode() \
        + png + f'\r\n--{b}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n--{b}--\r\n'.encode()
    return json.loads(call('/upload/image', body, f'multipart/form-data; boundary={b}'))['name']


# one fixed 3:4 box around everything she does in 0..186
union = np.max([cv2.imread(f'{W}/mask-BEN2-100_380_620_1080/%04d.png' % (k + 1), cv2.IMREAD_GRAYSCALE) for k in range(LAST + 1)], axis=0)
ys, xs = np.nonzero(union > 60)
h = int(max(ys.max() - ys.min(), (xs.max() - xs.min()) * 4 / 3) * 1.08)
w = h * 3 // 4
cx, cy = BX + (xs.min() + xs.max()) // 2, BY + (ys.min() + ys.max()) // 2
x0, y0 = cx - w // 2, cy - h // 2
print('drive box', x0, y0, w, h)
idx = [round(i * SRC_FPS / FPS) for i in range(int(LAST * FPS / SRC_FPS) + 1)]
if not os.environ.get('FETCH'):
    jobs = {}
    for i, k in enumerate(idx):
        f = cv2.imread(f'{W}/clean/%04d.png' % (k + 1))
        pad = cv2.copyMakeBorder(f, h, h, w, w, cv2.BORDER_REPLICATE)
        crop = cv2.resize(pad[y0 + h:y0 + 2 * h, x0 + w:x0 + 2 * w], (GW, GH), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'{D}/rgb/%04d.png' % i, crop)
        up = upload('bb_drive_%04d.png' % i, cv2.imencode('.png', crop)[1].tobytes())
        g = {'1': {'class_type': 'LoadImage', 'inputs': {'image': up}},
             '2': {'class_type': 'DWPreprocessor', 'inputs': {'image': ['1', 0], 'detect_hand': 'enable', 'detect_body': 'enable', 'detect_face': 'disable',
                                                              'resolution': GH, 'bbox_detector': 'yolox_l.torchscript.pt', 'pose_estimator': 'dw-ll_ucoco_384_bs5.torchscript.pt'}},
             '3': {'class_type': 'SaveImage', 'inputs': {'images': ['2', 0], 'filename_prefix': 'bootybounce/pose/p_%04d' % i}},
             '4': {'class_type': 'SavePoseKpsAsJsonFile', 'inputs': {'pose_kps': ['2', 1], 'filename_prefix': 'bootybounce/pose/kp_%04d' % i}}}
        jobs[json.loads(call('/prompt', json.dumps({'prompt': g}).encode()))['prompt_id']] = i
    print(len(jobs), 'frames queued', flush=True)
    while json.loads(call('/queue'))['queue_running'] or json.loads(call('/queue'))['queue_pending']:
        time.sleep(3)
# fetch skeleton image and keypoints of every frame (names as ComfyUI writes them)
for i in range(len(idx)):
    q = urllib.parse.urlencode({'filename': 'p_%04d_00001_.png' % i, 'subfolder': 'bootybounce/pose', 'type': 'output'})
    img = cv2.imdecode(np.frombuffer(call('/view?' + q), np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(f'{D}/pose/%04d.png' % i, cv2.resize(img, (GW, GH), interpolation=cv2.INTER_NEAREST))
    q = urllib.parse.urlencode({'filename': 'kp_%04d_00001.json' % i, 'subfolder': 'bootybounce/pose', 'type': 'output'})
    open(f'{D}/kp/%04d.json' % i, 'wb').write(call('/view?' + q))

# face crops from the head keypoints (nose, eyes, ears, neck), smoothed so the crop does not jitter
heads = []
for i in range(len(idx)):
    kp = json.load(open(f'{D}/kp/%04d.json' % i))
    kp = kp[0] if isinstance(kp, list) else kp
    p = np.array(kp['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
    sx, sy = GW / kp['canvas_width'], GH / kp['canvas_height']
    if p[:, :2].max() <= 1.0:  # normalised coordinates
        sx, sy = GW, GH
    pts = [(p[j, 0] * sx, p[j, 1] * sy) for j in (0, 14, 15, 16, 17) if p[j, 2] > 0.2]
    neck = p[1, :2] * (sx, sy)
    c = np.mean(pts, axis=0)
    heads.append([c[0], c[1], 2.4 * np.linalg.norm(c - neck)])
heads = np.array(heads)
smooth = np.array([heads[max(0, i - 2):i + 3].mean(axis=0) for i in range(len(heads))])
for i, (fx, fy, s) in enumerate(smooth):
    s = int(max(s, 60))
    rgb = cv2.imread(f'{D}/rgb/%04d.png' % i)
    pad = cv2.copyMakeBorder(rgb, s, s, s, s, cv2.BORDER_REPLICATE)
    face = pad[int(fy - s / 2) + s:int(fy - s / 2) + 2 * s, int(fx - s / 2) + s:int(fx - s / 2) + 2 * s]
    cv2.imwrite(f'{D}/face/%04d.png' % i, cv2.resize(face, (512, 512), interpolation=cv2.INTER_CUBIC))
print('face size px', int(smooth[:, 2].min()), int(smooth[:, 2].max()))
# check sheet: every 10th frame, rgb | pose | face
rows = []
for i in range(0, len(idx), 10):
    r = cv2.imread(f'{D}/rgb/%04d.png' % i); pz = cv2.imread(f'{D}/pose/%04d.png' % i)
    fc = cv2.resize(cv2.imread(f'{D}/face/%04d.png' % i), (GH // 2, GH // 2))
    fc = np.vstack([fc, np.zeros_like(fc)])
    t = np.hstack([r, pz, fc])
    rows.append(cv2.resize(t, None, fx=0.25, fy=0.25))
cv2.imwrite(f'{W}/debug/drive_sheet.png', np.vstack([np.hstack(rows[j:j + 2]) if j + 1 < len(rows) else np.hstack([rows[j], np.zeros_like(rows[j])]) for j in range(0, len(rows), 2)]))
print('frames', len(idx))
