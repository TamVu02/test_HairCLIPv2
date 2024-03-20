import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from scripts.Embedding_sg3 import Embedding_sg3
from scripts.text_proxy import TextProxy
from scripts.ref_proxy import RefProxy
from scripts.bald_proxy import BaldProxy
from scripts.feature_blending import hairstyle_feature_blending
from utils.seg_utils import vis_seg
from utils.mask_ui import painting_mask
from utils.image_utils import display_image_list, process_display_input
from utils.model_utils import load_sg3_models
from utils.options import Options

#Load args
opts = Options().parse()
src_name = '00008'# source image name you want to edit

#Load stylegan3 model for generator
image_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
generator, opts_sg3, mean_latent_code, seg = load_sg3_models(opts)

#Embedd source image into FS space
re4e = Embedding_sg3(opts, generator, mean_latent_code[0,0])
if not os.path.isfile(os.path.join(opts.src_latent_dir, f"{src_name}.npz")):
    inverted_latent_w_plus, inverted_latent_F = re4e.invert_image_in_FS(image_path=f'{opts.src_img_dir}/{src_name}.png')
    save_latent_path = os.path.join(opts.src_latent_dir, f'{src_name}.npz')
    np.savez(save_latent_path, latent_in=inverted_latent_w_plus.detach().cpu().numpy(),
                latent_F=inverted_latent_F.detach().cpu().numpy())

src_latent = torch.from_numpy(np.load(f'{opts.src_latent_dir}/{src_name}.npz')['latent_in']).cuda()
src_feature = torch.from_numpy(np.load(f'{opts.src_latent_dir}/{src_name}.npz')['latent_F']).cuda()
src_image = image_transform(Image.open(f'{opts.src_img_dir}/{src_name}.png').convert('RGB')).unsqueeze(0).cuda()
input_mask = torch.argmax(seg(src_image)[1], dim=1).long().clone().detach()

text_proxy = TextProxy(opts, generator, seg, mean_latent_code)
ref_proxy = RefProxy(opts, generator, seg, re4e)

latent_bald=None
if latent_bald is None:
    latent_bald, visual_global_list = text_proxy('bald hairstyle', src_image, from_mean=True,src_latent=src_latent)
np.savez('latent_bald.npy',latent_bald.detach().cpu().numpy())
