#!/usr/bin/env python3
# Step 3: the person in the new outfit doing the original motion. Wan 2.2 Animate 14B (GGUF Q4_K_M),
# lightx2v rank64 LoRA, clip_vision_h, umt5 Q5 GGUF, wan_2.1_vae; reference image + DWPose + face crops.
# usage: wan_animate.py OUTFIT [SEED]   -> WORK/wan/OUTFIT_sSEED.mp4 (16 fps, 576x768, two windows of 77 + 29 frames)
import cv2, json, numpy as np, os, subprocess, sys, time, urllib.parse, urllib.request, uuid
import imageio_ffmpeg
W = os.environ.get('WORK', 'work')  # make_pet.py work folder of the source clip
BASE = os.environ.get('COMFY', 'http://127.0.0.1:8188')
outfit = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 4242
GW, GH, FPS = 576, 768, 16
N = len(os.listdir(f'{W}/drive/pose'))
SEG1 = 77
SEG2 = ((N - SEG1 + 5) + 3) // 4 * 4 + 1  # 5 frames of motion carried over, length 4n+1
PROMPT = ('A young woman with long wavy blonde hair stands on the spot and stretches, raising and lowering her arm, '
          'against a plain light grey studio background. Realistic photo, natural motion, sharp detail, even soft light.')
NEG = 'blurry, low quality, distorted body, extra limbs, deformed hands, flicker, text, watermark, objects in the background'


def call(path, data=None, ctype='application/json'):
    r = urllib.request.Request(BASE + path, data=data, headers={'Content-Type': ctype})
    with urllib.request.urlopen(r, timeout=300) as f:
        return f.read()


def upload(name, png, sub=''):
    b = uuid.uuid4().hex
    parts = [f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\nContent-Type: image/png\r\n\r\n'.encode() + png,
             f'--{b}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue']
    if sub:
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="subfolder"\r\n\r\n{sub}')
    body = b'\r\n'.join(p if isinstance(p, bytes) else p.encode() for p in parts) + f'\r\n--{b}--\r\n'.encode()
    return json.loads(call('/upload/image', body, f'multipart/form-data; boundary={b}'))


for sub in ('pose', 'face'):
    for f in sorted(os.listdir(f'{W}/drive/{sub}')):
        upload(f, open(f'{W}/drive/{sub}/{f}', 'rb').read(), 'bb_' + sub)
ref = cv2.resize(cv2.imread(f'{W}/outfits/{outfit}.png'), (GW, GH), interpolation=cv2.INTER_AREA)
ref_name = upload('bb_ref_%s.png' % outfit.replace(' ', '_'), cv2.imencode('.png', ref)[1].tobytes())['name']

folder = lambda sub, w, h: {'class_type': 'LoadImagesFromFolderKJ', 'inputs': {'folder': os.environ.get('COMFY_INPUT', '/comfyui/input') + '/' + sub, 'width': w, 'height': h, 'keep_aspect_ratio': 'stretch',
                                                                               'image_load_cap': 0, 'start_index': 0, 'include_subfolders': False}}
g = {
    'unet': {'class_type': 'UnetLoaderGGUF', 'inputs': {'unet_name': 'Wan2.2-Animate-14B-Q4_K_M.gguf'}},
    'lora': {'class_type': 'LoraLoaderModelOnly', 'inputs': {'model': ['unet', 0], 'lora_name': 'lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors', 'strength_model': 1.0}},
    'shift': {'class_type': 'ModelSamplingSD3', 'inputs': {'model': ['lora', 0], 'shift': 5.0}},
    'clip': {'class_type': 'CLIPLoaderGGUF', 'inputs': {'clip_name': 'umt5-xxl-encoder-Q5_K_M.gguf', 'type': 'wan'}},
    'vae': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'wan_2.1_vae.safetensors'}},
    'cvl': {'class_type': 'CLIPVisionLoader', 'inputs': {'clip_name': 'clip_vision_h.safetensors'}},
    'ref': {'class_type': 'LoadImage', 'inputs': {'image': ref_name}},
    'cv': {'class_type': 'CLIPVisionEncode', 'inputs': {'clip_vision': ['cvl', 0], 'image': ['ref', 0], 'crop': 'none'}},
    'pos': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['clip', 0], 'text': PROMPT}},
    'neg': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['clip', 0], 'text': NEG}},
    'pose': folder('bb_pose', GW, GH),
    'face': folder('bb_face', 512, 512),
}
prev = None
for s, length in (('1', SEG1), ('2', SEG2)):
    a = {'positive': ['pos', 0], 'negative': ['neg', 0], 'vae': ['vae', 0], 'width': GW, 'height': GH, 'length': length, 'batch_size': 1,
         'continue_motion_max_frames': 5, 'video_frame_offset': 0 if prev is None else [f'anim{prev}', 5],
         'clip_vision_output': ['cv', 0], 'reference_image': ['ref', 0], 'face_video': ['face', 0], 'pose_video': ['pose', 0]}
    if prev:
        a['continue_motion'] = [f'img{prev}', 0]
    g[f'anim{s}'] = {'class_type': 'WanAnimateToVideo', 'inputs': a}
    g[f'ks{s}'] = {'class_type': 'KSampler', 'inputs': {'model': ['shift', 0], 'positive': [f'anim{s}', 0], 'negative': [f'anim{s}', 1],
                                                       'latent_image': [f'anim{s}', 2], 'seed': seed, 'steps': 6, 'cfg': 1.0,
                                                       'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1.0}}
    g[f'trim{s}'] = {'class_type': 'TrimVideoLatent', 'inputs': {'samples': [f'ks{s}', 0], 'trim_amount': [f'anim{s}', 3]}}
    g[f'dec{s}'] = {'class_type': 'VAEDecode', 'inputs': {'samples': [f'trim{s}', 0], 'vae': ['vae', 0]}}
    g[f'img{s}'] = {'class_type': 'ImageFromBatch', 'inputs': {'image': [f'dec{s}', 0], 'batch_index': [f'anim{s}', 4], 'length': 4096}}
    prev = s
g['all'] = {'class_type': 'ImageBatch', 'inputs': {'image1': ['img1', 0], 'image2': ['img2', 0]}}
tag = outfit.replace(' ', '_')
g['save'] = {'class_type': 'SaveImage', 'inputs': {'images': ['all', 0], 'filename_prefix': f'bootybounce/wan_{tag}_s{seed}/f'}}
t0 = time.time()
pid = json.loads(call('/prompt', json.dumps({'prompt': g}).encode()))['prompt_id']
print('queued', outfit, 'seed', seed, 'pose frames', N, 'segments', SEG1, SEG2, flush=True)
while True:
    time.sleep(10)
    h = json.loads(call('/history/' + pid)).get(pid)
    if not h:
        continue
    if h.get('status', {}).get('status_str') == 'error':
        sys.exit('error: ' + json.dumps(h['status'])[:3000])
    ims = [m for o in h.get('outputs', {}).values() for m in o.get('images', [])]
    if ims:
        break
out = f'{W}/wan/{tag}_s{seed}'
os.makedirs(out, exist_ok=True)
for k, m in enumerate(ims):
    q = urllib.parse.urlencode({x: m.get(x, '') for x in ('filename', 'subfolder', 'type')})
    open(f'{out}/%04d.png' % k, 'wb').write(call('/view?' + q))
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y', '-framerate', str(FPS), '-i', f'{out}/%04d.png',
                '-c:v', 'libx264', '-qp', '0', '-pix_fmt', 'yuv444p', f'{W}/wan/{tag}_s{seed}.mp4'], check=True)
print('%s: %d frames in %.1f min -> %s.mp4' % (outfit, len(ims), (time.time() - t0) / 60, out), flush=True)
