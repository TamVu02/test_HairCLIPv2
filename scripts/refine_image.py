import os
import numpy as np
import torch
import face_alignment
from utils.image_utils import process_display_input
import torch.nn.functional as F
from tqdm import tqdm
from criteria.transfer_loss import TransferLossBuilder

class Refine_image(torch.nn.Module):
    def __init__(self, opts, generator, seg):
        super(Refine_image, self).__init__()
        self.opts = opts
        self.generator = generator
        self.seg = seg
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

    def inference_on_kp_extractor(self, input_image):
        return self.kp_extractor.face_alignment_net(((F.interpolate(input_image, size=(256, 256)) + 1) / 2).clamp(0, 1))

    def gen_256_img_hairmask(self, input_image): 
        input_seg = torch.argmax(self.seg(input_image)[0].clone().detach(), dim=1).long()
        input_hairmask = torch.where((input_seg == 10), torch.ones_like(input_seg), torch.zeros_like(input_seg))
        input_hairmask_256 = F.interpolate(input_hairmask.unsqueeze(0).float(), size=(256, 256))
        input_img_256 = F.interpolate(input_image, size=(256, 256))
        return input_img_256, input_hairmask_256
    
    def gen_256_img_facemask(self, input_image): 
        input_seg = torch.argmax(self.seg(input_image)[0].clone().detach(), dim=1).long()
        input_hairmask = torch.where((input_seg == 10), torch.ones_like(input_seg), torch.zeros_like(input_seg))
        input_earmask = torch.where((input_seg == 6), torch.ones_like(input_seg), torch.zeros_like(input_seg))
        hair_ear_mask = input_hairmask + input_earmask
        input_face_256 = 1 - (F.interpolate(hair_ear_mask.unsqueeze(0).float(), size=(256, 256)))
        input_img_256 = F.interpolate(input_image, size=(256, 256))
        return input_img_256, input_face_256

    def forward(self, blended_latent, src_image, ref_img,m_style=6):
        ref_img_256, ref_hairmask_256 = self.gen_256_img_hairmask(ref_img)
        source_img_256, source_facemask_256 = self.gen_256_img_facemask(src_image)
        optimizer = torch.optim.Adam([blended_latent], lr=self.opts.lr_refine)
        latent_end = blended_latent[:, m_style:, :].clone().detach()
        latent_prev = blended_latent[:, :m_style, :].clone().detach()
        src_kp = self.inference_on_kp_extractor(src_image).clone().detach()

        visual_list = []
        visual_interval = self.opts.steps_refine // self.opts.visual_num_ref
        pbar = tqdm(range(self.opts.steps_refine))
        for i in pbar:
            optimizer.zero_grad()
            latent_in = torch.cat([blended_latent[:, :m_style, :], latent_end], dim=1)
            img_gen = self.generator.synthesis(latent_in, noise_mode='const')
            #Hair loss
            img_gen_256_hair, gen_hairmask_256 = self.gen_256_img_hairmask(img_gen)
            hair_style_loss = self.transfer_loss_builder.style_loss(ref_img_256, img_gen_256_hair, mask1=ref_hairmask_256, mask2=gen_hairmask_256)
            #Face loss
            img_gen_256_face, gen_facemask_256 = self.gen_256_img_facemask(img_gen)
            face_loss = self.transfer_loss_builder.style_loss(source_img_256, img_gen_256_face, mask1=source_facemask_256, mask2=gen_facemask_256)

            delta_w_loss = self.delta_loss(blended_latent[:, :m_style, :], latent_prev)

            gen_kp = self.inference_on_kp_extractor(img_gen)
            kp_loss = self.landmark_loss(src_kp[:, :], gen_kp[:, :])

            loss = self.opts.style_lambda_ref * (hair_style_loss + face_loss) + self.opts.delta_w_lambda_ref * delta_w_loss + self.opts.landmark_lambda_ref * kp_loss
            
            latent_prev = blended_latent[:, :m_style, :].clone().detach()
            loss.backward()
            optimizer.step()
            pbar.set_description((f"ref_loss: {loss.item():.4f};"))
            if (i % visual_interval == 0) or (i == (self.opts.steps_ref-1)):
                with torch.no_grad():
                    img_gen = self.generator.synthesis(latent_in, noise_mode='const')
                    visual_list.append(process_display_input(img_gen))

        return img_gen, latent_in, visual_list