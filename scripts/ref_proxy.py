import os
import numpy as np
import torch
import face_alignment
from utils.image_utils import process_display_input
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from criteria.transfer_loss import TransferLossBuilder
from PIL import Image
class RefProxy(torch.nn.Module):
    def __init__(self, opts, generator, seg, re4e):
        super(RefProxy, self).__init__()
        self.opts = opts
        self.generator = generator
        self.seg = seg
        self.re4e = re4e
        self.mask_loss = self.weighted_ce_loss()
        self.transfer_loss_builder = TransferLossBuilder()
        self.delta_loss = torch.nn.MSELoss()
        self.landmark_loss = torch.nn.MSELoss()
        self.kp_extractor = self.load_kp_extractor()

    def weighted_ce_loss(self):
        weight_tmp = torch.zeros(16).cuda()
        weight_tmp[10] = 1
        weight_tmp[1] = 1
        weight_tmp[6] = 1
        weight_tmp[0] = 1
        return torch.nn.CrossEntropyLoss(weight=weight_tmp).cuda()

    def load_kp_extractor(self):
        kp_extractor = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, device='cuda')
        for param in kp_extractor.face_alignment_net.parameters():
            param.requires_grad = False
        return kp_extractor
    
    def convert_npy_code(self,latent):
        if latent.shape == (16, 512):
            latent = np.reshape(latent, (1, 16, 512))
        if latent.shape == (512,) or latent.shape == (1, 512):
            latent = np.reshape(latent, (1, 1, 512)).repeat(16, axis=1)
        return latent
    
    def get_avg_img(self,generator):
        avg_image = generator(generator.latent_avg.repeat(16, 1).unsqueeze(0).cuda(),
                              input_code=True,
                              return_latents=False)[0]
        return avg_image

    def load_hairstyle_ref(self, hairstyle_ref_name):
        image_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        hairstyle_img_path = f'{self.opts.ref_img_dir}/{hairstyle_ref_name}'
        ref_PIL = Image.open(hairstyle_img_path).convert('RGB')
        ref_img = image_transform(ref_PIL).unsqueeze(0).cuda()

        print(os.path.join(self.opts.latents_path, f"{os.path.splitext(hairstyle_ref_name)[0]}.npy"))
        if not os.path.isfile(os.path.join(self.opts.latents_path, f"{os.path.splitext(hairstyle_ref_name)[0]}.npy")):
            inverted_wplus_code = self.re4e.invert_image_in_W(image_path=hairstyle_img_path)
            save_latent = inverted_wplus_code.detach().cpu().numpy()
            save_latent_path = os.path.join(self.opts.latents_path, f'{os.path.splitext(hairstyle_ref_name)[0]}.npy')
            np.save(save_latent_path, save_latent)

        latent_W_optimized = torch.from_numpy(self.convert_npy_code(np.load(os.path.join(self.opts.latents_path, f"{os.path.splitext(hairstyle_ref_name)[0]}.npy")))).cuda().requires_grad_(True)
        return ref_img, latent_W_optimized

    def inference_on_kp_extractor(self, input_image):
        return self.kp_extractor.face_alignment_net(((F.interpolate(input_image, size=(256, 256)) + 1) / 2).clamp(0, 1))

    def gen_256_img_hairmask(self, input_image): 
        input_seg = torch.argmax(self.seg(input_image)[0].clone().detach(), dim=1).long()
        input_hairmask = torch.where((input_seg == 10), torch.ones_like(input_seg), torch.zeros_like(input_seg))
        input_hairmask_256 = F.interpolate(input_hairmask.unsqueeze(0).float(), size=(256, 256))
        input_img_256 = F.interpolate(input_image, size=(256, 256))
        return input_img_256, input_hairmask_256

    def forward(self, hairstyle_ref_name, src_image, painted_mask=None):
        ref_img, latent_W_optimized = self.load_hairstyle_ref(hairstyle_ref_name)
        ref_img_256, ref_hairmask_256 = self.gen_256_img_hairmask(ref_img)
        ref_img_256 = ref_img_256.to('cuda')
        optimizer = torch.optim.Adam([latent_W_optimized], lr=self.opts.lr_ref)
        latent_end = latent_W_optimized[:, 4:, :].clone().detach()
        latent_prev = latent_W_optimized[:, :4, :].clone().detach()
        src_kp = self.inference_on_kp_extractor(src_image).clone().detach()

        visual_list = []
        visual_interval = self.opts.steps_ref // self.opts.visual_num_ref
        pbar = tqdm(range(self.opts.steps_ref))
        img_gen=None
        for i in pbar:
            optimizer.zero_grad()
            latent_in = torch.cat([latent_W_optimized[:, :4, :], latent_end], dim=1)
            if(i==0):
                avg_image = self.get_avg_img(self.generator)
                avg_image = avg_image.unsqueeze(0).repeat(ref_img_256.shape[0], 1, 1, 1)
                x_input = torch.cat([ref_img_256, avg_image], dim=1)
            else:
                x_input = torch.cat([ref_img_256, img_gen], dim=1)
            img_gen,_ = self.generator(x_input,latent=latent_in, return_latents=True, resize=False)
            img_gen_256, gen_hairmask_256 = self.gen_256_img_hairmask(img_gen)
            hair_style_loss = self.transfer_loss_builder.style_loss(ref_img_256, img_gen_256, mask1=ref_hairmask_256, mask2=gen_hairmask_256)

            delta_w_loss = self.delta_loss(latent_W_optimized[:, :4, :], latent_prev)

            gen_kp = self.inference_on_kp_extractor(img_gen)
            kp_loss = self.landmark_loss(src_kp[:, :], gen_kp[:, :])

            loss = self.opts.style_lambda_ref * hair_style_loss + self.opts.delta_w_lambda_ref * delta_w_loss + self.opts.landmark_lambda_ref * kp_loss

            if painted_mask is not None:
                down_seg = self.seg(img_gen)[1]
                hair_mask_loss = self.mask_loss(down_seg, painted_mask)
                loss += self.opts.hair_mask_lambda_ref * hair_mask_loss
            
            latent_prev = latent_W_optimized[:, :4, :].clone().detach()
            loss.backward()
            optimizer.step()
            pbar.set_description((f"ref_loss: {loss.item():.4f};"))
            if (i % visual_interval == 0) or (i == (self.opts.steps_ref-1)):
                with torch.no_grad():
                    img_gen = self.generator.decoder.synthesis(latent_in, noise_mode='const')
                    visual_list.append(process_display_input(img_gen))
            img_gen = self.generator.face_pool(img_gen).detach().clone()
        return latent_in, visual_list