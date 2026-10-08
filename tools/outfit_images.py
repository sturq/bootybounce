#!/usr/bin/env python3
# Step 1 of a new outfit: frame 1 of the person on plain grey, outfit changed with Krea 2 Identity Edit.
# Needs the Krea 2 base (lustify), the krea2_identity_edit LoRA and the comfyui-krea2edit nodes.
# The box and mask folder below are the ones make_pet.py used for Mint/Original.
import cv2, json, numpy as np, os, sys, time, urllib.parse, urllib.request, uuid
W = os.environ.get('WORK', 'work')  # make_pet.py work folder of the source clip
BASE = os.environ.get('COMFY', 'http://127.0.0.1:8188')
X0, Y0, X1, Y1 = 100, 380, 620, 1080
SEED = 4242
KEEP = ' Her hands are empty. Keep her face, hair, body, pose, framing and the plain grey background exactly the same.'
OUTFITS = {
    'Gym': 'Change her outfit to a black sports bra, black high-waisted gym leggings and white sneakers.',
    'Denim': 'Change her outfit to a white cropped t-shirt, light blue high-waisted jeans and white sneakers.',
    'Red Dress': 'Change her outfit to a short red satin dress and white sneakers.',
    'Hoodie': 'Change her outfit to a cropped pink hoodie, grey sweatpants and white sneakers.',
}


def call(path, data=None, ctype='application/json'):
    r = urllib.request.Request(BASE + path, data=data, headers={'Content-Type': ctype})
    with urllib.request.urlopen(r, timeout=120) as f:
        return f.read()


def upload(name, png):
    b = uuid.uuid4().hex
    body = (f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\nContent-Type: image/png\r\n\r\n').encode() \
        + png + f'\r\n--{b}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n--{b}--\r\n'.encode()
    return json.loads(call('/upload/image', body, f'multipart/form-data; boundary={b}'))['name']


OUT = f'{W}/outfits'
os.makedirs(OUT, exist_ok=True)
# reference: the cut-out person of frame 1 on plain grey, as Wan Animate later keeps the reference background
a = cv2.imread(f'{W}/mask-BEN2-100_380_620_1080/0001.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)[..., None] / 255
rgb = cv2.imread(f'{W}/clean/0001.png')[Y0:Y1, X0:X1].astype(np.float32)
ref = (rgb * a + 190 * (1 - a)).astype(np.uint8)
ys, xs = np.nonzero(a[..., 0] > 0.1)
h = int((ys.max() - ys.min()) * 1.12)
w = h * 3 // 4
cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
pad = cv2.copyMakeBorder(ref, h, h, w, w, cv2.BORDER_CONSTANT, value=(190, 190, 190))
ref = pad[cy + h - h // 2:cy + h - h // 2 + h, cx + w - w // 2:cx + w - w // 2 + w]
cv2.imwrite(f'{OUT}/reference.png', ref)
img = upload('bb_outfit_ref.png', cv2.imencode('.png', ref)[1].tobytes())
jobs = {}
for name, prompt in OUTFITS.items():
    g = {
        '1': {'class_type': 'LoadImage', 'inputs': {'image': img}},
        '2': {'class_type': 'ImageScaleToTotalPixels', 'inputs': {'image': ['1', 0], 'upscale_method': 'lanczos', 'megapixels': 1.0, 'resolution_steps': 16}},
        '3': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'lustifyNSFWCheckpoint_v10Krea2.safetensors', 'weight_dtype': 'default'}},
        '4': {'class_type': 'LoraLoaderModelOnly', 'inputs': {'model': ['3', 0], 'lora_name': 'krea2_identity_edit_v1_2.safetensors', 'strength_model': 1.0}},
        '5': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'qwen3vl_4b_bf16.safetensors', 'type': 'krea2', 'device': 'default'}},
        '6': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'qwen_image_vae.safetensors'}},
        '7': {'class_type': 'VAEEncode', 'inputs': {'pixels': ['2', 0], 'vae': ['6', 0]}},
        '8': {'class_type': 'GetImageSize', 'inputs': {'image': ['2', 0]}},
        '9': {'class_type': 'EmptySD3LatentImage', 'inputs': {'width': ['8', 0], 'height': ['8', 1], 'batch_size': 1}},
        '10': {'class_type': 'Krea2EditModelPatch', 'inputs': {'model': ['4', 0], 'source_latent': ['7', 0], 'ref_boost': 4.0, 'ref_boost_a': 1.0,
                                                                'fit_mode': 'fit', 'vae': ['6', 0], 'source_image': ['2', 0], 'target_latent': ['9', 0]}},
        '11': {'class_type': 'Krea2EditGroundedEncode', 'inputs': {'clip': ['5', 0], 'prompt': prompt + KEEP, 'image': ['2', 0], 'grounding_px': 768, 'system_prompt': ''}},
        '12': {'class_type': 'Krea2EditGroundedEncode', 'inputs': {'clip': ['5', 0], 'prompt': '', 'image': ['2', 0], 'grounding_px': 768, 'system_prompt': ''}},
        '13': {'class_type': 'KSampler', 'inputs': {'model': ['10', 0], 'positive': ['11', 0], 'negative': ['12', 0], 'latent_image': ['9', 0],
                                                    'seed': SEED, 'steps': 10, 'cfg': 1.0, 'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1.0}},
        '14': {'class_type': 'VAEDecode', 'inputs': {'samples': ['13', 0], 'vae': ['6', 0]}},
        '15': {'class_type': 'SaveImage', 'inputs': {'images': ['14', 0], 'filename_prefix': 'bootybounce/outfit_' + name.replace(' ', '_')}},
    }
    jobs[json.loads(call('/prompt', json.dumps({'prompt': g}).encode()))['prompt_id']] = name
t0 = time.time()
while jobs:
    time.sleep(5)
    for pid, name in list(jobs.items()):
        h = json.loads(call('/history/' + pid)).get(pid)
        if not h:
            continue
        if h.get('status', {}).get('status_str') == 'error':
            sys.exit('error %s: %s' % (name, json.dumps(h['status'])[:1500]))
        ims = [i for o in h.get('outputs', {}).values() for i in o.get('images', [])]
        if ims:
            q = urllib.parse.urlencode({k: ims[0].get(k, '') for k in ('filename', 'subfolder', 'type')})
            open(f'{OUT}/{name}.png', 'wb').write(call('/view?' + q))
            print('%s done after %.1f min' % (name, (time.time() - t0) / 60), flush=True)
            del jobs[pid]
# labelled sheet: reference + every outfit, same seed
tiles = []
for name in ['reference'] + list(OUTFITS):
    t = cv2.imread(f'{OUT}/{name}.png')
    t = cv2.resize(t, (int(t.shape[1] * 700 / t.shape[0]), 700), interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (t.shape[1], 34), (30, 30, 30), -1)
    cv2.putText(t, name + ('' if name == 'reference' else '  seed %d' % SEED), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
    tiles.append(t)
cv2.imwrite(f'{OUT}/sheet.png', np.hstack(tiles))
