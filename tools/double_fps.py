#!/usr/bin/env python3
# Step 4: 16 to 32 fps with ComfyUI's own frame interpolation. FILM keeps fast arms whole, plain optical
# flow drew them twice. Models: Comfy-Org/frame_interpolation (film_net_fp16, rife_v4.26).
# usage: double_fps.py RENDER MODEL   e.g. double_fps.py Gym_s4242 film_net_fp16.safetensors
import cv2, json, os, subprocess, sys, time, urllib.parse, urllib.request, uuid
import imageio_ffmpeg
W = os.environ.get('WORK', 'work')  # make_pet.py work folder of the source clip
name, model = sys.argv[1], sys.argv[2]
BASE = os.environ.get('COMFY', 'http://127.0.0.1:8188')
src = f'{W}/wan/{name}'
tag = model.split('.')[0]
out = f'{W}/wan/{name}_x2_{tag}'


def call(path, data=None, ctype='application/json'):
    r = urllib.request.Request(BASE + path, data=data, headers={'Content-Type': ctype})
    with urllib.request.urlopen(r, timeout=300) as f:
        return f.read()


def upload(fname, png, sub):
    b = uuid.uuid4().hex
    body = (f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="{fname}"\r\nContent-Type: image/png\r\n\r\n').encode() + png + \
        (f'\r\n--{b}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n--{b}\r\nContent-Disposition: form-data; name="subfolder"\r\n\r\n{sub}\r\n--{b}--\r\n').encode()
    call('/upload/image', body, f'multipart/form-data; boundary={b}')


sub = 'bb_wan_' + name
files = sorted(os.listdir(src))
for f in files:
    upload(f, open(f'{src}/{f}', 'rb').read(), sub)
h, w = cv2.imread(f'{src}/{files[0]}').shape[:2]
g = {'load': {'class_type': 'LoadImagesFromFolderKJ', 'inputs': {'folder': os.environ.get('COMFY_INPUT', '/comfyui/input') + '/' + sub, 'width': w, 'height': h, 'keep_aspect_ratio': 'stretch',
                                                                  'image_load_cap': 0, 'start_index': 0, 'include_subfolders': False}},
     'm': {'class_type': 'FrameInterpolationModelLoader', 'inputs': {'model_name': model}},
     'fi': {'class_type': 'FrameInterpolate', 'inputs': {'interp_model': ['m', 0], 'images': ['load', 0], 'multiplier': 2}},
     'save': {'class_type': 'SaveImage', 'inputs': {'images': ['fi', 0], 'filename_prefix': f'bootybounce/{name}_x2_{tag}/f'}}}
t0 = time.time()
pid = json.loads(call('/prompt', json.dumps({'prompt': g}).encode()))['prompt_id']
while True:
    time.sleep(5)
    hst = json.loads(call('/history/' + pid)).get(pid)
    if not hst:
        continue
    if hst.get('status', {}).get('status_str') == 'error':
        sys.exit('error: ' + json.dumps(hst['status'])[:2000])
    ims = [m for o in hst.get('outputs', {}).values() for m in o.get('images', [])]
    if ims:
        break
os.makedirs(out, exist_ok=True)
for k, m in enumerate(ims):
    q = urllib.parse.urlencode({x: m.get(x, '') for x in ('filename', 'subfolder', 'type')})
    open(f'{out}/%04d.png' % k, 'wb').write(call('/view?' + q))
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y', '-framerate', '32', '-i', f'{out}/%04d.png',
                '-c:v', 'libx264', '-qp', '0', '-pix_fmt', 'yuv444p', out + '.mp4'], check=True)
print('%s %s: %d -> %d frames in %.1f min' % (name, model, len(files), len(ims), (time.time() - t0) / 60))
