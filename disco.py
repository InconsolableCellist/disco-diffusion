import gc
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from glob import glob
from pathlib import Path
from types import SimpleNamespace
import cv2
import lpips
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from IPython import display
from PIL import Image, ImageOps
from ipywidgets import Output
from torch import nn
from torch.nn import functional as F
from tqdm.notebook import tqdm
from CLIP import clip

PROJECT_DIR = os.path.abspath(os.getcwd())
USE_ADABINS = True
ROOT_PATH = os.getcwd()

def createPath(filepath):
    os.makedirs(filepath, exist_ok=True)

ROOT_PATH = os.getcwd()
MODEL_PATH = f'{ROOT_PATH}/models'

INIT_DIR_PATH = f'{ROOT_PATH}/init_images'
createPath(INIT_DIR_PATH)
OUT_DIR_PATH = f'{ROOT_PATH}/images_out'

createPath(OUT_DIR_PATH)

MODEL_PATH = f'{ROOT_PATH}/models'
createPath(MODEL_PATH)

def gitclone(url):
    res = subprocess.run(['git', 'clone', url], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)

def wget(url, outputdir):
    res = subprocess.run(['wget', url, '-P', f'{outputdir}'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)

try:
    from CLIP import clip
except:
    if not os.path.exists("CLIP"):
        gitclone("https://github.com/openai/CLIP")
    sys.path.append(f'{PROJECT_DIR}/CLIP')
    from CLIP import clip

try:
    from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
except:
    if not os.path.exists("guided-diffusion"):
        gitclone("https://github.com/kostarion/guided-diffusion")
    sys.path.append(f'{PROJECT_DIR}/guided-diffusion')
    from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

# try:
#     from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
# except:
#     if not os.path.exists("guided-diffusion"):
#         gitclone("https://github.com/crowsonkb/guided-diffusion")
#     sys.path.append(f'{PROJECT_DIR}/guided-diffusion')
#     from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

try:
    from resize_right import resize
except:
    if not os.path.exists("ResizeRight"):
        gitclone("https://github.com/assafshocher/ResizeRight.git")
    sys.path.append(f'{PROJECT_DIR}/ResizeRight')
    from resize_right import resize

try:
    import py3d_tools
except:
    if not os.path.exists('pytorch3d-lite'):
        gitclone("https://github.com/MSFTserver/pytorch3d-lite.git")
    sys.path.append(f'{PROJECT_DIR}/pytorch3d-lite')
    import py3d_tools

try:
    from midas.dpt_depth import DPTDepthModel
except:
    if not os.path.exists('MiDaS'):
        gitclone("https://github.com/isl-org/MiDaS.git")
    if not os.path.exists('MiDaS/midas_utils.py'):
        shutil.move('MiDaS/utils.py', 'MiDaS/midas_utils.py')
    if not os.path.exists(f'{MODEL_PATH}/dpt_large-midas-2f21e586.pt'):
        wget("https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt", MODEL_PATH)
    sys.path.append(f'{PROJECT_DIR}/MiDaS')
    from midas.dpt_depth import DPTDepthModel



nvidiasmi_output = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8')
print(nvidiasmi_output)
nvidiasmi_ecc_note = subprocess.run(['nvidia-smi', '-i', '0', '-e', '0'], stdout=subprocess.PIPE).stdout.decode('utf-8')
print(nvidiasmi_ecc_note)



useCPU = False

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


multipip_res = subprocess.run(['pip', 'install', 'lpips', 'datetime', 'timm', 'ftfy', 'einops', 'pytorch-lightning', 'omegaconf'], stdout=subprocess.PIPE).stdout.decode('utf-8')
print(multipip_res)


os.chdir(f'{PROJECT_DIR}')
warnings.filterwarnings("ignore", category=UserWarning)

if USE_ADABINS:
    try:
        from infer import InferenceHelper
    except:
        if os.path.exists("AdaBins") is not True:
            gitclone("https://github.com/shariqfarooq123/AdaBins.git")
        if not os.path.exists(f'{PROJECT_DIR}/pretrained/AdaBins_nyu.pt'):
            createPath(f'{PROJECT_DIR}/pretrained')
            wget("https://cloudflare-ipfs.com/ipfs/Qmd2mMnDLWePKmgfS8m6ntAg4nhV5VkUyAydYBp8cWWeB7/AdaBins_nyu.pt", f'{PROJECT_DIR}/pretrained')
        sys.path.append(f'{PROJECT_DIR}/AdaBins')
    MAX_ADABINS_AREA = 500000

try:
    sys.path.append(PROJECT_DIR)
    import disco_xform_utils as dxf
except:
    if not os.path.exists("disco-diffusion"):
        gitclone("https://github.com/alembics/disco-diffusion.git")
    if os.path.exists('disco_xform_utils.py') is not True:
        shutil.move('disco-diffusion/disco_xform_utils.py', 'disco_xform_utils.py')
    sys.path.append(PROJECT_DIR)
    import disco_xform_utils as dxf


DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and not useCPU) else 'cpu')
print('Using device:', DEVICE)
device = DEVICE

if not useCPU:
    if torch.cuda.get_device_capability(DEVICE) == (8,0): ## A100 fix thanks to Emad
        print('Disabling CUDNN for A100 gpu', file=sys.stderr)
        torch.backends.cudnn.enabled = False

# Initialize MiDaS depth model.
# It remains resident in VRAM and likely takes around 2GB VRAM.
# You could instead initialize it for each frame (and free it after each frame) to save VRAM.. but initializing it is slow.
default_models = {
    "midas_v21_small": f"{MODEL_PATH}/midas_v21_small-70d6b9c8.pt",
    "midas_v21": f"{MODEL_PATH}/midas_v21-f6b98070.pt",
    "dpt_large": f"{MODEL_PATH}/dpt_large-midas-2f21e586.pt",
    "dpt_hybrid": f"{MODEL_PATH}/dpt_hybrid-midas-501f0c75.pt",
    "dpt_hybrid_nyu": f"{MODEL_PATH}/dpt_hybrid_nyu-2ce69ec7.pt",}

def init_midas_depth_model(midas_model_type="dpt_large", optimize=True):
    midas_model = None
    net_w = None
    net_h = None
    resize_mode = None
    normalization = None

    print(f"Initializing MiDaS '{midas_model_type}' depth model...")
    # load network
    midas_model_path = default_models[midas_model_type]

    if midas_model_type == "dpt_large": # DPT-Large
        midas_model = DPTDepthModel(
            path=midas_model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif midas_model_type == "dpt_hybrid": #DPT-Hybrid
        midas_model = DPTDepthModel(
            path=midas_model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode="minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif midas_model_type == "dpt_hybrid_nyu": #DPT-Hybrid-NYU
        midas_model = DPTDepthModel(
            path=midas_model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode="minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif midas_model_type == "midas_v21":
        midas_model = MidasNet(midas_model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif midas_model_type == "midas_v21_small":
        midas_model = MidasNet_small(midas_model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        print(f"midas_model_type '{midas_model_type}' not implemented")
        assert False

    midas_transform = T.Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    midas_model.eval()

    if optimize==True:
        if DEVICE == torch.device("cuda"):
            midas_model = midas_model.to(memory_format=torch.channels_last)
            midas_model = midas_model.half()

    midas_model.to(DEVICE)

    print(f"MiDaS '{midas_model_type}' depth model initialized.")
    return midas_model, midas_transform, net_w, net_h, resize_mode, normalization

def interp(t):
    return 3 * t**2 - 2 * t ** 3

def perlin(width, height, scale=10, device=None):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)

def perlin_ms(octaves, width, height, grayscale, device=device):
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    # out_array = [0.0] if grayscale else [0.0, 0.0, 0.0]
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale, device)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)

def create_perlin_noise(octaves=[1, 1, 1, 1], width=2, height=2, grayscale=True):
    out = perlin_ms(octaves, width, height, grayscale)
    if grayscale:
        out = TF.resize(size=(side_y, side_x), img=out.unsqueeze(0))
        out = TF.to_pil_image(out.clamp(0, 1)).convert('RGB')
    else:
        out = out.reshape(-1, 3, out.shape[0]//3, out.shape[1])
        out = TF.resize(size=(side_y, side_x), img=out)
        out = TF.to_pil_image(out.clamp(0, 1).squeeze())

    out = ImageOps.autocontrast(out)
    return out

def regen_perlin():
    if perlin_mode == 'color':
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, False)
    elif perlin_mode == 'gray':
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, True)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)
    else:
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)

    init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
    del init2
    return init.expand(batch_size, -1, -1, -1)

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

def read_image_workaround(path):
    """OpenCV reads images as BGR, Pillow saves them as RGB. Work around
    this incompatibility to avoid colour inversions."""
    im_tmp = cv2.imread(path)
    return cv2.cvtColor(im_tmp, cv2.COLOR_BGR2RGB)

def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])

def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))

def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()

def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]

def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomPerspective(distortion_scale=0.4, p=0.7),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.15),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def forward(self, input):
        input = T.Pad(input.shape[2]//4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn//4:
                cutout = input.clone()
            else:
                size = int(max_size * torch.zeros(1,).normal_(mean=.8, std=.3).clip(float(self.cut_size/max_size), 1.))
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        return cutouts

cutout_debug = False
padargs = {}

class MakeCutoutsDango(nn.Module):
    def __init__(self, cut_size,
                 Overview=4,
                 InnerCrop = 0, IC_Size_Pow=0.5, IC_Grey_P = 0.2
                 ):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05),  interpolation = T.InterpolationMode.BILINEAR),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.1),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])



    def forward(self, input):
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        l_size = max(sideX, sideY)
        output_shape = [1,3,self.cut_size,self.cut_size]
        output_shape_2 = [1,3,self.cut_size+2,self.cut_size+2]
        pad_input = F.pad(input,((sideY-max_size)//2,(sideY-max_size)//2,(sideX-max_size)//2,(sideX-max_size)//2), **padargs)
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview>0:
            if self.Overview<=4:
                if self.Overview>=1:
                    cutouts.append(cutout)
                if self.Overview>=2:
                    cutouts.append(gray(cutout))
                if self.Overview>=3:
                    cutouts.append(TF.hflip(cutout))
                if self.Overview==4:
                    cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

            if cutout_debug:
                TF.to_pil_image(cutouts[0].clamp(0, 1).squeeze(0)).save("cutout_overview0.jpg",quality=99)


        if self.InnerCrop >0:
            for i in range(self.InnerCrop):
                size = int(torch.rand([])**self.IC_Size_Pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
            if cutout_debug:
                TF.to_pil_image(cutouts[-1].clamp(0, 1).squeeze(0)).save("cutout_InnerCrop.jpg",quality=99)
        cutouts = torch.cat(cutouts)
        if skip_augs is not True: cutouts=self.augs(cutouts)
        return cutouts

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])

stop_on_next_loop = False  # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete
TRANSLATION_SCALE = 1.0/200.0

def symmetry_transformation_fn(x):
  if args.use_horizontal_symmetry:
    [n, c, h, w] = x.size()
    x = torch.concat((x[:, :, :, :w//2], torch.flip(x[:, :, :, :w//2], [-1])), -1)
    print("horizontal symmetry applied")
  if args.use_vertical_symmetry:
    [n, c, h, w] = x.size()
    x = torch.concat((x[:, :, :h//2, :], torch.flip(x[:, :, :h//2, :], [-2])), -2)
    print("vertical symmetry applied")
  return x


def do_run():
    seed = args.seed
    print(range(args.start_frame, args.max_frames))

    for frame_num in range(args.start_frame, args.max_frames):
        if stop_on_next_loop:
            break

        display.clear_output(wait=True)

        if args.init_image in ['','none', 'None', 'NONE']:
            init_image = None
        else:
            init_image = args.init_image
        init_scale = args.init_scale
        skip_steps = args.skip_steps

        loss_values = []

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        target_embeds, weights = [], []

        if args.prompts_series is not None and frame_num >= len(args.prompts_series):
            frame_prompt = args.prompts_series[-1]
        elif args.prompts_series is not None:
            frame_prompt = args.prompts_series[frame_num]
        else:
            frame_prompt = []

        print(args.image_prompts_series)
        if args.image_prompts_series is not None and frame_num >= len(args.image_prompts_series):
            image_prompt = args.image_prompts_series[-1]
        elif args.image_prompts_series is not None:
            image_prompt = args.image_prompts_series[frame_num]
        else:
            image_prompt = []

        print(f'Frame {frame_num} Prompt: {frame_prompt}')

        model_stats = []
        for clip_model in clip_models:
            cutn = 16
            model_stat = {"clip_model":None,"target_embeds":[],"make_cutouts":None,"weights":[]}
            model_stat["clip_model"] = clip_model


            for prompt in frame_prompt:
                txt, weight = parse_prompt(prompt)
                txt = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()

                if args.fuzzy_prompt:
                    for i in range(25):
                        model_stat["target_embeds"].append((txt + torch.randn(txt.shape).cuda() * args.rand_mag).clamp(0,1))
                        model_stat["weights"].append(weight)
                else:
                    model_stat["target_embeds"].append(txt)
                    model_stat["weights"].append(weight)

            if image_prompt:
                model_stat["make_cutouts"] = MakeCutouts(clip_model.visual.input_resolution, cutn, skip_augs=skip_augs)
                for prompt in image_prompt:
                    path, weight = parse_prompt(prompt)
                    img = Image.open(fetch(path)).convert('RGB')
                    img = TF.resize(img, min(side_x, side_y, *img.size), T.InterpolationMode.LANCZOS)
                    batch = model_stat["make_cutouts"](TF.to_tensor(img).to(device).unsqueeze(0).mul(2).sub(1))
                    embed = clip_model.encode_image(normalize(batch)).float()
                    if fuzzy_prompt:
                        for i in range(25):
                            model_stat["target_embeds"].append((embed + torch.randn(embed.shape).cuda() * rand_mag).clamp(0,1))
                            weights.extend([weight / cutn] * cutn)
                    else:
                        model_stat["target_embeds"].append(embed)
                        model_stat["weights"].extend([weight / cutn] * cutn)

            model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
            model_stat["weights"] = torch.tensor(model_stat["weights"], device=device)
            if model_stat["weights"].sum().abs() < 1e-3:
                raise RuntimeError('The weights must not sum to 0.')
            model_stat["weights"] /= model_stat["weights"].sum().abs()
            model_stats.append(model_stat)

        init = None
        if init_image is not None:
            init = Image.open(fetch(init_image)).convert('RGB')
            init = init.resize((args.side_x, args.side_y), Image.LANCZOS)
            init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

        if args.perlin_init:
            if args.perlin_mode == 'color':
                init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
                init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, False)
            elif args.perlin_mode == 'gray':
                init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, True)
                init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)
            else:
                init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
                init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)
            # init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device)
            init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
            del init2

        cur_t = None

        def cond_fn(x, t, y=None):
            with torch.enable_grad():
                x_is_NaN = False
                x = x.detach().requires_grad_()
                n = x.shape[0]
                if use_secondary_model is True:
                    alpha = torch.tensor(diffusion.sqrt_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                    sigma = torch.tensor(diffusion.sqrt_one_minus_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                    cosine_t = alpha_sigma_to_t(alpha, sigma)
                    out = secondary_model(x, cosine_t[None].repeat([n])).pred
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)
                else:
                    my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
                    out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out['pred_xstart'] * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)
                for model_stat in model_stats:
                    for i in range(args.cutn_batches):
                        t_int = int(t.item())+1 #errors on last step without +1, need to find source
                        #when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
                        try:
                            input_resolution=model_stat["clip_model"].visual.input_resolution
                        except:
                            input_resolution=224

                        cuts = MakeCutoutsDango(input_resolution,
                                                Overview= args.cut_overview[1000-t_int],
                                                InnerCrop = args.cut_innercut[1000-t_int], IC_Size_Pow=args.cut_ic_pow, IC_Grey_P = args.cut_icgray_p[1000-t_int]
                                                )
                        clip_in = normalize(cuts(x_in.add(1).div(2)))
                        image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
                        dists = spherical_dist_loss(image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0))
                        dists = dists.view([args.cut_overview[1000-t_int]+args.cut_innercut[1000-t_int], n, -1])
                        losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                        loss_values.append(losses.sum().item()) # log loss, probably shouldn't do per cutn_batch
                        x_in_grad += torch.autograd.grad(losses.sum() * clip_guidance_scale, x_in)[0] / cutn_batches
                tv_losses = tv_loss(x_in)
                if use_secondary_model is True:
                    range_losses = range_loss(out)
                else:
                    range_losses = range_loss(out['pred_xstart'])
                sat_losses = torch.abs(x_in - x_in.clamp(min=-1,max=1)).mean()
                loss = tv_losses.sum() * tv_scale + range_losses.sum() * range_scale + sat_losses.sum() * sat_scale
                if init is not None and args.init_scale:
                    init_losses = lpips_model(x_in, init)
                    loss = loss + init_losses.sum() * args.init_scale
                x_in_grad += torch.autograd.grad(loss, x_in)[0]
                if torch.isnan(x_in_grad).any()==False:
                    grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
                else:
                    # print("NaN'd")
                    x_is_NaN = True
                    grad = torch.zeros_like(x)
            if args.clamp_grad and x_is_NaN == False:
                magnitude = grad.square().mean().sqrt()
                return grad * magnitude.clamp(max=args.clamp_max) / magnitude  #min=-0.02, min=-clamp_max,
            return grad

        if args.diffusion_sampling_mode == 'ddim':
            sample_fn = diffusion.ddim_sample_loop_progressive
        else:
            sample_fn = diffusion.plms_sample_loop_progressive


        image_display = Output()
        for i in range(args.n_batches):
            display.clear_output(wait=True)
            batchBar = tqdm(range(args.n_batches), desc ="Batches")
            batchBar.n = i
            batchBar.refresh()
            print('')
            display.display(image_display)
            gc.collect()
            torch.cuda.empty_cache()
            cur_t = diffusion.num_timesteps - skip_steps - 1
            total_steps = cur_t

            if perlin_init:
                init = regen_perlin()

            if args.diffusion_sampling_mode == 'ddim':
                samples = sample_fn(
                    model,
                    (batch_size, 3, args.side_y, args.side_x),
                    clip_denoised=clip_denoised,
                    model_kwargs={},
                    cond_fn=cond_fn,
                    progress=True,
                    skip_timesteps=skip_steps,
                    init_image=init,
                    randomize_class=randomize_class,
                    eta=eta,
                    transformation_fn=symmetry_transformation_fn,
                    transformation_percent=args.transformation_steps
                )
            else:
                samples = sample_fn(
                    model,
                    (batch_size, 3, args.side_y, args.side_x),
                    clip_denoised=clip_denoised,
                    model_kwargs={},
                    cond_fn=cond_fn,
                    progress=True,
                    skip_timesteps=skip_steps,
                    init_image=init,
                    randomize_class=randomize_class,
                    order=2,
                )


            # with run_display:
            # display.clear_output(wait=True)
            for j, sample in enumerate(samples):
                cur_t -= 1
                intermediateStep = False
                if args.steps_per_checkpoint is not None:
                    if j % steps_per_checkpoint == 0 and j > 0:
                        intermediateStep = True
                elif j in args.intermediate_saves:
                    intermediateStep = True
                with image_display:
                    if j % args.display_rate == 0 or cur_t == -1 or intermediateStep == True:
                        for k, image in enumerate(sample['pred_xstart']):
                            # tqdm.write(f'Batch {i}, step {j}, output {k}:')
                            current_time = datetime.now().strftime('%y%m%d-%H%M%S_%f')
                            percent = math.ceil(j/total_steps*100)
                            if args.n_batches > 0:
                                #if intermediates are saved to the subfolder, don't append a step or percentage to the name
                                if cur_t == -1 and args.intermediates_in_subfolder is True:
                                    save_num = i
                                    filename = f'{batch_name}({batchNum})_{save_num}.png'
                                else:
                                    #If we're working with percentages, append it
                                    if args.steps_per_checkpoint is not None:
                                        filename = f'{batch_name}({batchNum})_{i:04}-{percent:02}%.png'
                                    # Or else, iIf we're working with specific steps, append those
                                    else:
                                        filename = f'{batch_name}({batchNum})_{i:04}-{j:03}.png'
                            image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                            if j % args.display_rate == 0 or cur_t == -1:
                                image.save('progress.png')
                                display.clear_output(wait=True)
                                display.display(display.Image('progress.png'))
                            if args.steps_per_checkpoint is not None:
                                if j % args.steps_per_checkpoint == 0 and j > 0:
                                    if args.intermediates_in_subfolder is True:
                                        image.save(f'{partialFolder}/{filename}')
                                    else:
                                        image.save(f'{batchFolder}/{filename}')
                            else:
                                if j in args.intermediate_saves:
                                    if args.intermediates_in_subfolder is True:
                                        image.save(f'{partialFolder}/{filename}')
                                    else:
                                        image.save(f'{batchFolder}/{filename}')
                            if cur_t == -1:
                                if frame_num == 0:
                                    save_settings()
                                print(f'want to save to {batchFolder}/{filename}')
                                image.save(f'{batchFolder}/{filename}')

            plt.plot(np.array(loss_values), 'r')

def save_settings():
    try:
        with open(f"{batchFolder}/{batch_name}({batchNum})_settings.txt", "w+", encoding='utf-8') as f:
            json.dump(args_saved, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f'Error saving settings: {e}')

def append_dims(x, n):
    return x[(Ellipsis, *(None,) * (n - x.ndim))]


def expand_to_planes(x, shape):
    return append_dims(x, len(shape)).repeat([1, 1, *shape[2:]])


def alpha_sigma_to_t(alpha, sigma):
    return torch.atan2(sigma, alpha) * 2 / math.pi


def t_to_alpha_sigma(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


@dataclass
class DiffusionOutput:
    v: torch.Tensor
    pred: torch.Tensor
    eps: torch.Tensor


class ConvBlock(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class SecondaryDiffusionImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count

        self.timestep_embed = FourierFeatures(1, 16)

        self.net = nn.Sequential(
            ConvBlock(3 + 16, c),
            ConvBlock(c, c),
            SkipBlock([
                nn.AvgPool2d(2),
                ConvBlock(c, c * 2),
                ConvBlock(c * 2, c * 2),
                SkipBlock([
                    nn.AvgPool2d(2),
                    ConvBlock(c * 2, c * 4),
                    ConvBlock(c * 4, c * 4),
                    SkipBlock([
                        nn.AvgPool2d(2),
                        ConvBlock(c * 4, c * 8),
                        ConvBlock(c * 8, c * 4),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    ]),
                    ConvBlock(c * 8, c * 4),
                    ConvBlock(c * 4, c * 2),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ]),
                ConvBlock(c * 4, c * 2),
                ConvBlock(c * 2, c),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ]),
            ConvBlock(c * 2, c),
            nn.Conv2d(c, 3, 3, padding=1),
        )

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


class SecondaryDiffusionImageNet2(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8]

        self.timestep_embed = FourierFeatures(1, 16)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.net = nn.Sequential(
            ConvBlock(3 + 16, cs[0]),
            ConvBlock(cs[0], cs[0]),
            SkipBlock([
                self.down,
                ConvBlock(cs[0], cs[1]),
                ConvBlock(cs[1], cs[1]),
                SkipBlock([
                    self.down,
                    ConvBlock(cs[1], cs[2]),
                    ConvBlock(cs[2], cs[2]),
                    SkipBlock([
                        self.down,
                        ConvBlock(cs[2], cs[3]),
                        ConvBlock(cs[3], cs[3]),
                        SkipBlock([
                            self.down,
                            ConvBlock(cs[3], cs[4]),
                            ConvBlock(cs[4], cs[4]),
                            SkipBlock([
                                self.down,
                                ConvBlock(cs[4], cs[5]),
                                ConvBlock(cs[5], cs[5]),
                                ConvBlock(cs[5], cs[5]),
                                ConvBlock(cs[5], cs[4]),
                                self.up,
                            ]),
                            ConvBlock(cs[4] * 2, cs[4]),
                            ConvBlock(cs[4], cs[3]),
                            self.up,
                        ]),
                        ConvBlock(cs[3] * 2, cs[3]),
                        ConvBlock(cs[3], cs[2]),
                        self.up,
                    ]),
                    ConvBlock(cs[2] * 2, cs[2]),
                    ConvBlock(cs[2], cs[1]),
                    self.up,
                ]),
                ConvBlock(cs[1] * 2, cs[1]),
                ConvBlock(cs[1], cs[0]),
                self.up,
            ]),
            ConvBlock(cs[0] * 2, cs[0]),
            nn.Conv2d(cs[0], 3, 3, padding=1),
        )

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)

def split_prompts(prompts):
    prompt_series = pd.Series([np.nan for a in range(max_frames)])
    for i, prompt in prompts.items():
        prompt_series[i] = prompt
    # prompt_series = prompt_series.astype(str)
    prompt_series = prompt_series.ffill().bfill()
    return prompt_series

if len(sys.argv) < 2:
    print(f'Please specify the settings.json file')
    sys.exit()
settings_file = sys.argv[1]
args = {}
try:
    with open(settings_file) as f:
        args = json.load(f)
except Exception as e:
    print(f'Could not load settings file: {settings_file}. {e}')
    sys.exit()

if 'max_frames' not in args:
    args['max_frames'] = 1
if 'start_frame' not in args:
    args['start_frame'] = 0
if 'batchNum' not in args:
    args['batchNum'] = 0
if 'batch_name' not in args:
    args['batch_name'] = 'TimeToDisco'
if 'diffusion_steps' not in args:
    args['diffusion_steps'] = diffusion_steps
if 'display_rate' not in args:
    args['display_rate'] = 30
if 'batch_size' not in args:
    args['batch_size'] = 1
if 'key_frames' not in args:
    args['key_frames'] = True
if 'max_frames' not in args:
    args['max_frames'] = 10000
if 'init_scale' not in args:
    args['init_scale'] = 1000
if 'skip_steps' not in args:
    args['skip_steps'] = 20
if 'init_image' not in args:
    args['init_image'] = None
if 'steps_per_checkpoint' not in args:
    args['steps_per_checkpoint'] = 10
if 'set_seed' not in args:
    args['set_seed'] = 'random_seed'
if 'intermediate_saves' not in args:
    args['intermediate_saves'] = 0
if 'n_batches' not in args:
    args['n_batches'] = 1
if 'intermediates_in_subfolder' not in args:
    args['intermediates_in_subfolder'] = True
if 'use_vertical_symmetry' not in args:
    args['use_vertical_symmetry'] = False
if 'use_horizontal_symmetry' not in args:
    args['use_horizontal_symmetry'] = False
if 'transformation_steps' not in args:
    args['transformation_steps'] = [0.01]

width_height = [args['width'], args['height']]
side_x = (width_height[0]//64)*64;
side_y = (width_height[1]//64)*64;
if side_x != width_height[0] or side_y != width_height[1]:
    print(f'Changing output size to {side_x}x{side_y}. Dimensions must by multiples of 64.')

args['side_x'] = side_x
args['side_y'] = side_y
set_seed = args['set_seed']

ViTB32 = args['ViTB32']
RN50 = args['RN50']
RN101 = args['RN101']
ViTB16 = args['ViTB16']
RN50x4 = args['RN50x4']
RN50x16 = args['RN50x16']
RN50x64 = args['RN50x64']
ViTL14 = args['ViTL14']
ViTL14_336 = args['ViTL14_336']
diffusion_model = args['diffusion_model']
use_secondary_model = args['use_secondary_model']
diffusion_sampling_mode = args['diffusion_sampling_mode']

batch_name = args['batch_name']
batch_size = args['batch_size']
steps = args['steps']
steps_per_checkpoint = steps+10
args['steps_per_checkpoint'] = steps_per_checkpoint
clip_guidance_scale = args['clip_guidance_scale']
tv_scale = args['tv_scale']
range_scale = args['range_scale']
sat_scale = args['sat_scale']
cutn = args['cutn']
cutn_batches = args['cutn_batches']
skip_augs = args['skip_augs']
display_rate = args['display_rate']
key_frames = args['key_frames']
max_frames = args['max_frames']
init_scale = args['init_scale']
skip_steps = args['skip_steps']
init_image = args['init_image']
intermediate_saves = args['intermediate_saves']
intermediates_in_subfolder = args['intermediates_in_subfolder']
n_batches = args['n_batches']

args['prompts_series'] = split_prompts(args['text_prompts']) if 'text_prompts' in args else None
args['image_prompts_series'] = split_prompts(args['image_prompts']) if ('image_prompts' in args and args['image_prompts'] != {}) else None


perlin_init = args['perlin_init']
perlin_mode = args['perlin_mode']
eta = args['eta']
clamp_grad = args['clamp_grad']
clamp_max = args['clamp_max']
randomize_class = args['randomize_class']
clip_denoised = args['clip_denoised']
fuzzy_prompt = args['fuzzy_prompt']
rand_mag = args['rand_mag']

if intermediate_saves and intermediates_in_subfolder is True:
    partialFolder = f'{batchFolder}/partials'
    createPath(partialFolder)

animation_mode = 'None'
max_frames = 1
args['max_frames'] = max_frames

args_saved = args.copy()
del args_saved['prompts_series']

args['cut_overview'] = eval(args['cut_overview'])
args['cut_innercut'] = eval(args['cut_innercut'])
args['cut_icgray_p'] = eval(args['cut_icgray_p'])
args['cut_ic_pow'] = eval(str(args['cut_ic_pow']))

args = SimpleNamespace(**args)
use_checkpoint = True

check_model_SHA = False #@param{type:"boolean"}

def download_models(diffusion_model,use_secondary_model,fallback=False):
    model_256_downloaded = False
    model_512_downloaded = False
    model_secondary_downloaded = False

    model_256_SHA = '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a'
    model_512_SHA = '9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648'
    model_secondary_SHA = '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a'

    model_256_link = 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'
    model_512_link = 'https://v-diffusion.s3.us-west-2.amazonaws.com/512x512_diffusion_uncond_finetune_008100.pt'
    model_secondary_link = 'https://v-diffusion.s3.us-west-2.amazonaws.com/secondary_model_imagenet_2.pth'

    model_256_link_fb = 'https://www.dropbox.com/s/9tqnqo930mpnpcn/256x256_diffusion_uncond.pt'
    model_512_link_fb = 'https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt'
    model_secondary_link_fb = 'https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth'

    model_256_path = f'{MODEL_PATH}/256x256_diffusion_uncond.pt'
    model_512_path = f'{MODEL_PATH}/512x512_diffusion_uncond_finetune_008100.pt'
    model_secondary_path = f'{MODEL_PATH}/secondary_model_imagenet_2.pth'

    if fallback:
        model_256_link = model_256_link_fb
        model_512_link = model_512_link_fb
        model_secondary_link = model_secondary_link_fb
    # Download the diffusion model
    if diffusion_model == '256x256_diffusion_uncond':
        if os.path.exists(model_256_path) and check_model_SHA:
            print('Checking 256 Diffusion File')
            with open(model_256_path,"rb") as f:
                bytes = f.read()
                hash = hashlib.sha256(bytes).hexdigest();
            if hash == model_256_SHA:
                print('256 Model SHA matches')
                model_256_downloaded = True
            else:
                print("256 Model SHA doesn't match, redownloading...")
                wget(model_256_link, MODEL_PATH)
                if os.path.exists(model_256_path):
                    model_256_downloaded = True
                else:
                    print('First URL Failed using FallBack')
                    download_models(diffusion_model,use_secondary_model,True)
        elif os.path.exists(model_256_path) and not check_model_SHA or model_256_downloaded == True:
            print('256 Model already downloaded, check check_model_SHA if the file is corrupt')
        else:
            wget(model_256_link, MODEL_PATH)
            if os.path.exists(model_256_path):
                model_256_downloaded = True
            else:
                print('First URL Failed using FallBack')
                download_models(diffusion_model,True)
    elif diffusion_model == '512x512_diffusion_uncond_finetune_008100':
        if os.path.exists(model_512_path) and check_model_SHA:
            print('Checking 512 Diffusion File')
            with open(model_512_path,"rb") as f:
                bytes = f.read()
                hash = hashlib.sha256(bytes).hexdigest();
            if hash == model_512_SHA:
                print('512 Model SHA matches')
                if os.path.exists(model_512_path):
                    model_512_downloaded = True
                else:
                    print('First URL Failed using FallBack')
                    download_models(diffusion_model,use_secondary_model,True)
            else:
                print("512 Model SHA doesn't match, redownloading...")
                wget(model_512_link, MODEL_PATH)
                if os.path.exists(model_512_path):
                    model_512_downloaded = True
                else:
                    print('First URL Failed using FallBack')
                    download_models(diffusion_model,use_secondary_model,True)
        elif os.path.exists(model_512_path) and not check_model_SHA or model_512_downloaded == True:
            print('512 Model already downloaded, check check_model_SHA if the file is corrupt')
        else:
            wget(model_512_link, MODEL_PATH)
            model_512_downloaded = True
    # Download the secondary diffusion model v2
    if use_secondary_model == True:
        if os.path.exists(model_secondary_path) and check_model_SHA:
            print('Checking Secondary Diffusion File')
            with open(model_secondary_path,"rb") as f:
                bytes = f.read()
                hash = hashlib.sha256(bytes).hexdigest();
            if hash == model_secondary_SHA:
                print('Secondary Model SHA matches')
                model_secondary_downloaded = True
            else:
                print("Secondary Model SHA doesn't match, redownloading...")
                wget(model_secondary_link, MODEL_PATH)
                if os.path.exists(model_secondary_path):
                    model_secondary_downloaded = True
                else:
                    print('First URL Failed using FallBack')
                    download_models(diffusion_model,use_secondary_model,True)
        elif os.path.exists(model_secondary_path) and not check_model_SHA or model_secondary_downloaded == True:
            print('Secondary Model already downloaded, check check_model_SHA if the file is corrupt')
        else:
            wget(model_secondary_link, MODEL_PATH)
            if os.path.exists(model_secondary_path):
                model_secondary_downloaded = True
            else:
                print('First URL Failed using FallBack')
                download_models(diffusion_model,use_secondary_model,True)

download_models(diffusion_model,use_secondary_model,True)

model_config = model_and_diffusion_defaults()
if diffusion_model == '512x512_diffusion_uncond_finetune_008100':
    model_config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': 1000, #No need to edit this, it is taken care of later.
        'rescale_timesteps': True,
        'timestep_respacing': 250, #No need to edit this, it is taken care of later.
        'image_size': 512,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_checkpoint': use_checkpoint,
        'use_fp16': not useCPU,
        'use_scale_shift_norm': True,
    })
elif diffusion_model == '256x256_diffusion_uncond':
    model_config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': 1000, #No need to edit this, it is taken care of later.
        'rescale_timesteps': True,
        'timestep_respacing': 250, #No need to edit this, it is taken care of later.
        'image_size': 256,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_checkpoint': use_checkpoint,
        'use_fp16': not useCPU,
        'use_scale_shift_norm': True,
    })

model_default = model_config['image_size']



if use_secondary_model:
    secondary_model = SecondaryDiffusionImageNet2()
    secondary_model.load_state_dict(torch.load(f'{MODEL_PATH}/secondary_model_imagenet_2.pth', map_location='cpu'))
    secondary_model.eval().requires_grad_(False).to(device)

clip_models = []
if ViTB32 is True: clip_models.append(clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device))
if ViTB16 is True: clip_models.append(clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(device) )
if ViTL14 is True: clip_models.append(clip.load('ViT-L/14', jit=False)[0].eval().requires_grad_(False).to(device) )
if RN50 is True: clip_models.append(clip.load('RN50', jit=False)[0].eval().requires_grad_(False).to(device))
if RN50x4 is True: clip_models.append(clip.load('RN50x4', jit=False)[0].eval().requires_grad_(False).to(device))
if RN50x16 is True: clip_models.append(clip.load('RN50x16', jit=False)[0].eval().requires_grad_(False).to(device))
if RN50x64 is True: clip_models.append(clip.load('RN50x64', jit=False)[0].eval().requires_grad_(False).to(device))
if RN101 is True: clip_models.append(clip.load('RN101', jit=False)[0].eval().requires_grad_(False).to(device))
if ViTL14_336 is True: clip_models.append(clip.load('ViT-L/14@336px', jit=False)[0].eval().requires_grad_(False).to(device))

normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
lpips_model = lpips.LPIPS(net='vgg').to(device)

#Update Model Settings
timestep_respacing = f'ddim{steps}'
diffusion_steps = (1000//steps)*steps if steps < 1000 else steps
model_config.update({
    'timestep_respacing': timestep_respacing,
    'diffusion_steps': diffusion_steps,
})

batchFolder = f'{OUT_DIR_PATH}/{batch_name}'
createPath(batchFolder)


video_init_path = "training.mp4"
extract_nth_frame = 2
video_init_seed_continuity = True

# key_frames = True
# max_frames = 10000

interp_spline = 'Linear' #Do not change, currently will not look good. param ['Linear','Quadratic','Cubic']{type:"string"}
angle = "0:(0)"
zoom = "0: (1), 10: (1.05)"
translation_x = "0: (0)"
translation_y = "0: (0)"
translation_z = "0: (10.0)"
rotation_3d_x = "0: (0)"
rotation_3d_y = "0: (0)"
rotation_3d_z = "0: (0)"
midas_depth_model = "dpt_large"
midas_weight = 0.3
near_plane = 200
far_plane = 10000
fov = 40
padding_mode = 'border'
sampling_mode = 'bicubic'

turbo_mode = False #@param {type:"boolean"}
turbo_steps = "3" #@param ["2","3","4","5","6"] {type:"string"}
turbo_preroll = 10 # frames

#@markdown ---

#@markdown ####**Coherency Settings:**
#@markdown `frame_scale` tries to guide the new frame to looking like the old one. A good default is 1500.
frames_scale = 1500 #@param{type: 'integer'}
#@markdown `frame_skip_steps` will blur the previous frame - higher values will flicker less but struggle to add enough new detail to zoom into.
frames_skip_steps = '60%' #@param ['40%', '50%', '60%', '70%', '80%'] {type: 'string'}


def parse_key_frames(string, prompt_parser=None):
    """Given a string representing frame numbers paired with parameter values at that frame,
    return a dictionary with the frame numbers as keys and the parameter values as the values.

    Parameters
    ----------
    string: string
        Frame numbers paired with parameter values at that frame number, in the format
        'framenumber1: (parametervalues1), framenumber2: (parametervalues2), ...'
    prompt_parser: function or None, optional
        If provided, prompt_parser will be applied to each string of parameter values.
    
    Returns
    -------
    dict
        Frame numbers as keys, parameter values at that frame number as values

    Raises
    ------
    RuntimeError
        If the input string does not match the expected format.
    
    Examples
    --------
    >>> parse_key_frames("10:(Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)")
    {10: 'Apple: 1| Orange: 0', 20: 'Apple: 0| Orange: 1| Peach: 1'}

    >>> parse_key_frames("10:(Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)", prompt_parser=lambda x: x.lower()))
    {10: 'apple: 1| orange: 0', 20: 'apple: 0| orange: 1| peach: 1'}
    """
    pattern = r'((?P<frame>[0-9]+):[\s]*[\(](?P<param>[\S\s]*?)[\)])'
    frames = dict()
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()['frame'])
        param = match_object.groupdict()['param']
        if prompt_parser:
            frames[frame] = prompt_parser(param)
        else:
            frames[frame] = param

    if frames == {} and len(string) != 0:
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames

def get_inbetweens(key_frames, integer=False):
    """Given a dict with frame numbers as keys and a parameter value as values,
    return a pandas Series containing the value of the parameter at every frame from 0 to max_frames.
    Any values not provided in the input dict are calculated by linear interpolation between
    the values of the previous and next provided frames. If there is no previous provided frame, then
    the value is equal to the value of the next provided frame, or if there is no next provided frame,
    then the value is equal to the value of the previous provided frame. If no frames are provided,
    all frame values are NaN.

    Parameters
    ----------
    key_frames: dict
        A dict with integer frame numbers as keys and numerical values of a particular parameter as values.
    integer: Bool, optional
        If True, the values of the output series are converted to integers.
        Otherwise, the values are floats.
    
    Returns
    -------
    pd.Series
        A Series with length max_frames representing the parameter values for each frame.
    
    Examples
    --------
    >>> max_frames = 5
    >>> get_inbetweens({1: 5, 3: 6})
    0    5.0
    1    5.0
    2    5.5
    3    6.0
    4    6.0
    dtype: float64

    >>> get_inbetweens({1: 5, 3: 6}, integer=True)
    0    5
    1    5
    2    5
    3    6
    4    6
    dtype: int64
    """
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])

    for i, value in key_frames.items():
        key_frame_series[i] = value
    key_frame_series = key_frame_series.astype(float)

    interp_method = interp_spline

    if interp_method == 'Cubic' and len(key_frames.items()) <=3:
        interp_method = 'Quadratic'

    if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
        interp_method = 'Linear'


    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames-1] = key_frame_series[key_frame_series.last_valid_index()]
    # key_frame_series = key_frame_series.interpolate(method=intrp_method,order=1, limit_direction='both')
    key_frame_series = key_frame_series.interpolate(method=interp_method.lower(),limit_direction='both')
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series


if key_frames:
    try:
        angle_series = get_inbetweens(parse_key_frames(angle))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `angle` correctly for key frames.\n"
            "Attempting to interpret `angle` as "
            f'"0: ({angle})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        angle = f"0: ({angle})"
        angle_series = get_inbetweens(parse_key_frames(angle))

    try:
        zoom_series = get_inbetweens(parse_key_frames(zoom))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `zoom` correctly for key frames.\n"
            "Attempting to interpret `zoom` as "
            f'"0: ({zoom})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        zoom = f"0: ({zoom})"
        zoom_series = get_inbetweens(parse_key_frames(zoom))

    try:
        translation_x_series = get_inbetweens(parse_key_frames(translation_x))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `translation_x` correctly for key frames.\n"
            "Attempting to interpret `translation_x` as "
            f'"0: ({translation_x})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        translation_x = f"0: ({translation_x})"
        translation_x_series = get_inbetweens(parse_key_frames(translation_x))

    try:
        translation_y_series = get_inbetweens(parse_key_frames(translation_y))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `translation_y` correctly for key frames.\n"
            "Attempting to interpret `translation_y` as "
            f'"0: ({translation_y})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        translation_y = f"0: ({translation_y})"
        translation_y_series = get_inbetweens(parse_key_frames(translation_y))

    try:
        translation_z_series = get_inbetweens(parse_key_frames(translation_z))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `translation_z` correctly for key frames.\n"
            "Attempting to interpret `translation_z` as "
            f'"0: ({translation_z})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        translation_z = f"0: ({translation_z})"
        translation_z_series = get_inbetweens(parse_key_frames(translation_z))

    try:
        rotation_3d_x_series = get_inbetweens(parse_key_frames(rotation_3d_x))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `rotation_3d_x` correctly for key frames.\n"
            "Attempting to interpret `rotation_3d_x` as "
            f'"0: ({rotation_3d_x})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        rotation_3d_x = f"0: ({rotation_3d_x})"
        rotation_3d_x_series = get_inbetweens(parse_key_frames(rotation_3d_x))

    try:
        rotation_3d_y_series = get_inbetweens(parse_key_frames(rotation_3d_y))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `rotation_3d_y` correctly for key frames.\n"
            "Attempting to interpret `rotation_3d_y` as "
            f'"0: ({rotation_3d_y})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        rotation_3d_y = f"0: ({rotation_3d_y})"
        rotation_3d_y_series = get_inbetweens(parse_key_frames(rotation_3d_y))

    try:
        rotation_3d_z_series = get_inbetweens(parse_key_frames(rotation_3d_z))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `rotation_3d_z` correctly for key frames.\n"
            "Attempting to interpret `rotation_3d_z` as "
            f'"0: ({rotation_3d_z})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        rotation_3d_z = f"0: ({rotation_3d_z})"
        rotation_3d_z_series = get_inbetweens(parse_key_frames(rotation_3d_z))

else:
    angle = float(angle)
    zoom = float(zoom)
    translation_x = float(translation_x)
    translation_y = float(translation_y)
    translation_z = float(translation_z)
    rotation_3d_x = float(rotation_3d_x)
    rotation_3d_y = float(rotation_3d_y)
    rotation_3d_z = float(rotation_3d_z)



#Update Model Settings
timestep_respacing = f'ddim{steps}'
diffusion_steps = (1000//steps)*steps if steps < 1000 else steps
model_config.update({
    'timestep_respacing': timestep_respacing,
    'diffusion_steps': diffusion_steps,
})

def move_files(start_num, end_num, old_folder, new_folder):
    for i in range(start_num, end_num):
        old_file = old_folder + f'/{batch_name}({batchNum})_{i:04}.png'
        new_file = new_folder + f'/{batch_name}({batchNum})_{i:04}.png'
        os.rename(old_file, new_file)

#@markdown ---


resume_run = False #@param{type: 'boolean'}
run_to_resume = 'latest' #@param{type: 'string'}
resume_from_frame = 'latest' #@param{type: 'string'}
retain_overwritten_frames = False #@param{type: 'boolean'}
if retain_overwritten_frames is True:
    retainFolder = f'{batchFolder}/retained'
    createPath(retainFolder)


skip_step_ratio = int(frames_skip_steps.rstrip("%")) / 100
calc_frames_skip_steps = math.floor(steps * skip_step_ratio)


if steps <= calc_frames_skip_steps:
    sys.exit("ERROR: You can't skip more steps than your total steps")

if resume_run:
    if run_to_resume == 'latest':
        try:
            batchNum
        except:
            batchNum = len(glob(f"{batchFolder}/{batch_name}(*)_settings.txt"))-1
    else:
        batchNum = int(run_to_resume)
    if resume_from_frame == 'latest':
        start_frame = len(glob(batchFolder+f"/{batch_name}({batchNum})_*.png"))
        if animation_mode != '3D' and turbo_mode == True and start_frame > turbo_preroll and start_frame % int(turbo_steps) != 0:
            start_frame = start_frame - (start_frame % int(turbo_steps))
    else:
        start_frame = int(resume_from_frame)+1
        if animation_mode != '3D' and turbo_mode == True and start_frame > turbo_preroll and start_frame % int(turbo_steps) != 0:
            start_frame = start_frame - (start_frame % int(turbo_steps))
        if retain_overwritten_frames is True:
            existing_frames = len(glob(batchFolder+f"/{batch_name}({batchNum})_*.png"))
            frames_to_save = existing_frames - start_frame
            print(f'Moving {frames_to_save} frames to the Retained folder')
            move_files(start_frame, existing_frames, batchFolder, retainFolder)
else:
    start_frame = 0
    batchNum = len(glob(batchFolder+"/*.txt"))
    while os.path.isfile(f"{batchFolder}/{batch_name}({batchNum})_settings.txt") is True or os.path.isfile(f"{batchFolder}/{batch_name}-{batchNum}_settings.txt") is True:
        batchNum += 1

print(f'Starting Run: {batch_name}({batchNum}) at frame {start_frame}')

if set_seed == 'random_seed':
    random.seed()
    seed = random.randint(0, 2**32)
    args_saved['seed'] = seed
    args.seed = seed
    # print(f'Using seed: {seed}')
else:
    seed = int(set_seed)


print('Prepping model...')
model, diffusion = create_model_and_diffusion(**model_config)
model.load_state_dict(torch.load(f'{MODEL_PATH}/{diffusion_model}.pt', map_location='cpu'))
model.requires_grad_(False).eval().to(device)
for name, param in model.named_parameters():
    if 'qkv' in name or 'norm' in name or 'proj' in name:
        param.requires_grad_()
if model_config['use_fp16']:
    model.convert_to_fp16()

gc.collect()
torch.cuda.empty_cache()
try:
    do_run()
except KeyboardInterrupt:
    pass
finally:
    print('Seed used:', seed)
    gc.collect()
    torch.cuda.empty_cache()
