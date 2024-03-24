import torch
from utils.image_utils import dliate_erode
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from scripts.text_proxy import TextProxy
from utils.common import tensor2im
from utils.inference_utils import get_average_image

img_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def hairstyle_feature_blending(generator, seg, src_image, visual_mask, latent_bald, latent_global=None):

    if latent_global is not None:
        bald_feature = generator.decoder.synthesis(latent_bald, noise_mode='const')
        global_feature = generator.decoder.synthesis(latent_global, noise_mode='const')
        global_proxy = generator.decoder.synthesis(latent_global, noise_mode='const')
        global_proxy_seg = torch.argmax(seg(global_proxy)[1], dim=1).unsqueeze(1).long()

        ear_mask = torch.where(visual_mask==6, torch.ones_like(visual_mask), torch.zeros_like(visual_mask))[0].cpu().numpy()
        hair_mask = torch.where(visual_mask==10, torch.ones_like(visual_mask), torch.zeros_like(visual_mask))[0].cpu().numpy()
        hair_ear_mask = ear_mask + hair_mask
        bald_blending_mask = dliate_erode(hair_ear_mask.astype('uint8'), 30)
        bald_blending_mask = torch.from_numpy(bald_blending_mask).unsqueeze(0).unsqueeze(0).cuda()
        bald_blending_mask_down = F.interpolate(bald_blending_mask.float(), size=(1024, 1024), mode='bicubic')
        src_image = bald_feature * bald_blending_mask_down + src_image * (1-bald_blending_mask_down)

        global_hair_mask = torch.where(global_proxy_seg==10, torch.ones_like(global_proxy_seg), torch.zeros_like(global_proxy_seg))
        global_hair_mask_down = F.interpolate(global_hair_mask.float(), size=(1024, 1024), mode='bicubic')
        src_image = global_feature * global_hair_mask_down + src_image * (1-global_hair_mask_down)

    feat_out_img = tensor2im(src_image[-1])
    out = img_transforms(feat_out_img).unsqueeze(0).to('cuda')

    with torch.no_grad():
            avg_image = get_average_image(generator)
            avg_image = avg_image.unsqueeze(0).repeat(out.shape[0], 1, 1, 1)
            x_input = torch.cat([out, avg_image], dim=1)
            img_gen_blend,blend_latent = generator(x_input,latent=None, return_latents=True, resize=False)
    return feat_out_img, img_gen_blend, blend_latent