import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from scripts.Embedding_sg3 import Embedding_sg3
from scripts.text_proxy import TextProxy
from scripts.ref_proxy import RefProxy
from scripts.feature_blending import hairstyle_feature_blending
from utils.seg_utils import vis_seg
from utils.mask_ui import painting_mask
from utils.image_utils import display_image_list, process_display_input
from utils.model_utils import load_sg3_models
from utils.options import Options

#Load args
opts = Options().parse()
src_name = '168125'# source image name you want to edit

#Load stylegan3 model for generator
image_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
generator, opts_sg3, mean_latent_code, seg = load_sg3_models(opts)

#Embedd source image into FS space
re4e = Embedding_sg3(opts, generator, mean_latent_code[0,0])
if not os.path.isfile(os.path.join(opts.src_latent_dir, f"{src_name}.npz")):
    inverted_latent_w_plus, inverted_latent_F = re4e.invert_image_in_FS(image_path=f'{opts.src_img_dir}/{src_name}.jpg')
    save_latent_path = os.path.join(opts.src_latent_dir, f'{src_name}.npz')
    np.savez(save_latent_path, latent_in=inverted_latent_w_plus.detach().cpu().numpy(),
                latent_F=inverted_latent_F.detach().cpu().numpy())
    
src_latent = torch.from_numpy(np.load(f'{opts.src_latent_dir}/{src_name}.npz')['latent_in']).cuda()
src_feature = torch.from_numpy(np.load(f'{opts.src_latent_dir}/{src_name}.npz')['latent_F']).cuda()
src_image = image_transform(Image.open(f'{opts.src_img_dir}/{src_name}.jpg').convert('RGB')).unsqueeze(0).cuda()
input_mask = torch.argmax(seg(src_image)[1], dim=1).long().clone().detach()

text_proxy = TextProxy(opts, generator, seg, mean_latent_code)
ref_proxy = RefProxy(opts, generator, seg, re4e)

edited_hairstyle_img = src_image
def hairstyle_editing(global_cond=None, paint_the_mask=False, \
                      src_latent=src_latent, src_feature=src_feature, input_mask=input_mask, src_image=src_image, \
                        latent_global=None, latent_local=None, local_blending_mask=None, painted_mask=None):
    if paint_the_mask:
        modified_mask = painting_mask(input_mask)
        input_mask = torch.from_numpy(modified_mask).unsqueeze(0).cuda().long().clone().detach()
        vis_modified_mask = vis_seg(modified_mask)
        display_image_list([src_image, vis_modified_mask])
        painted_mask = input_mask
        
    if global_cond is not None:
        if global_cond.endswith('.jpg') or global_cond.endswith('.png'):
            latent_global, visual_global_list = ref_proxy(global_cond, src_image, painted_mask=painted_mask)
        else:
            latent_global, visual_global_list = text_proxy(global_cond, src_image, from_mean=True, painted_mask=painted_mask)
        display_image_list(visual_global_list)

    src_feature, edited_hairstyle_img = hairstyle_feature_blending(generator, seg, src_latent, src_feature, input_mask,\
                                                latent_global=latent_global, latent_local=latent_local, local_blending_mask=local_blending_mask)
    return src_feature, edited_hairstyle_img


