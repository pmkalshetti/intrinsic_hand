import torch
import numpy as np
import nvdiffrast.torch as dr
import cv2 as cv
import plotly.graph_objects as go
import time
import os
from torchviz import make_dot, make_dot_from_trace
from lpips import LPIPS
from tqdm import tqdm
from pytorch3d.transforms.so3 import (
    so3_exp_map,
    so3_log_map,
)

from ..mano.mano import Mano
from ..render import optixutils as ou
from ..render import renderutils as ru
from ..render import bilateral_denoiser, light, texture, render, regularizer, util, mesh, material, obj
from ..metrics.ssim import calculate_ssim,  calculate_msssim
from .. import utils

logger = utils.get_logger(__name__)

def create_trainable_mat(cfg):
    kd_min, kd_max = torch.tensor(cfg.MAT.KD_MIN, dtype=torch.float32, device=cfg.DEVICE), torch.tensor(cfg.MAT.KD_MAX, dtype=torch.float32, device=cfg.DEVICE)
    
    kd_init = texture.srgb_to_rgb(texture.load_texture2D(cfg.MANO.HTML_KD, channels=3))
    kd_map_opt = texture.create_trainable(kd_init , cfg.MAT.TEXTURE_RES, True, [kd_min, kd_max])

    ks_min, ks_max = torch.tensor(cfg.MAT.KS_MIN, dtype=torch.float32, device=cfg.DEVICE), torch.tensor(cfg.MAT.KS_MAX, dtype=torch.float32, device=cfg.DEVICE)
    ksR = np.random.uniform(size=cfg.MAT.TEXTURE_RES + [1], low=0.0, high=0.01)
    ksG = np.random.uniform(size=cfg.MAT.TEXTURE_RES + [1], low=ks_min[1].cpu(), high=ks_max[1].cpu())
    ksB = np.random.uniform(size=cfg.MAT.TEXTURE_RES + [1], low=ks_min[2].cpu(), high=ks_max[2].cpu())

    ks_map_opt = texture.create_trainable(np.concatenate((ksR, ksG, ksB), axis=2), cfg.MAT.TEXTURE_RES, True, [ks_min, ks_max])
    
    nrm_min, nrm_max = torch.tensor(cfg.MAT.NRM_MIN, dtype=torch.float32, device=cfg.DEVICE), torch.tensor(cfg.MAT.NRM_MAX, dtype=torch.float32, device=cfg.DEVICE)
    normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), cfg.MAT.TEXTURE_RES, True, [nrm_min, nrm_max])

    mat = {
        'kd'     : kd_map_opt,
        'ks'     : ks_map_opt,
        'normal' : normal_map_opt,
        'bsdf'   : cfg.MAT.BSDF,
        'no_perturbed_nrm': cfg.MAT.NO_PERTURBED_NRM
    }

    return mat

@torch.no_grad()
def mix_background(batch):
    # Mix background into a dataset image
    background = torch.zeros(batch['img'].shape[0:3] + (3,), dtype=torch.float32, device=batch['img'].device)
    batch['background'] = background
    batch['img'] = torch.cat((torch.lerp(background, batch['img'][..., 0:3], batch['img'][..., 3:4]), batch['img'][..., 3:4]), dim=-1)

    return batch

class IHAOptimizer:
    def __init__(self, cfg, dataset):
        self.cfg = cfg
        self.dataset = dataset
        self.num_data = len(self.dataset)
        # ==============================================================================================
        #  Create trainable mesh (with fixed topology)
        # ==============================================================================================
        self.mano = Mano(self.cfg.MANO).to(self.cfg.DEVICE)
        
        self.beta = torch.zeros(10, requires_grad=True, device=self.cfg.DEVICE)
        self.offsets = torch.zeros(len(self.mano.v_template), 3, requires_grad=True, device=self.cfg.DEVICE)
        self.list_hand_pose = [torch.zeros(15*3, requires_grad=True, device=self.cfg.DEVICE) for _ in range(self.num_data)]
        if self.cfg.DATA.NAME == "kinect" or "syn" in self.cfg.DATA.NAME:
            self.list_global_rot = [torch.zeros(3, requires_grad=True, device=self.cfg.DEVICE) for _ in range(self.num_data)]
            self.list_global_transl = [torch.zeros(3, requires_grad=True, device=self.cfg.DEVICE) for _ in range(self.num_data)]
        elif self.cfg.DATA.NAME == "interhand":
            self.list_global_orient = [torch.zeros(3, requires_grad=True, device=self.cfg.DEVICE) for _ in range(self.num_data)]
            self.list_transl = [torch.zeros(3, requires_grad=True, device=self.cfg.DEVICE) for _ in range(self.num_data)]
        

        # ==============================================================================================
        #  Create trainable material
        # ==============================================================================================
        self.mat = create_trainable_mat(self.cfg)

        # ==============================================================================================
        #  Create trainable light
        # ==============================================================================================
        self.lgt = light.create_trainable_env_rnd(self.cfg.RENDER.PROBE_RES, scale=0.0, bias=0.5)
        
        # ==============================================================================================
        #  Setup denoiser
        # ==============================================================================================
        self.denoiser = bilateral_denoiser.BilateralDenoiser().to(self.cfg.DEVICE)
        
        self.glctx = dr.RasterizeGLContext(device=self.cfg.DEVICE) # Context for training
        with torch.no_grad():
            self.optix_ctx = ou.OptiXContext()
            self.image_loss_fn = lambda img, ref: ru.image_loss(img, ref, loss="l1", tonemapper="log_srgb")

        self.setup_optimizer()
        
        self.base_mesh = mesh.Mesh(
            v_pos=self.mano.v_template, t_pos_idx=self.mano.faces,
            v_tex=self.mano.verts_uvs, t_tex_idx=self.mano.faces_uvs,
            material=self.mat
        )
        self.base_mesh = mesh.auto_normals(self.base_mesh)

        self.lpips = LPIPS(net='vgg').to(self.cfg.DEVICE)

    def setup_optimizer(self):
        def lr_schedule(iter, fraction):
            warmup_iter = 0
            if iter < warmup_iter:
                return iter / warmup_iter 
            return max(0.0, 10**(-(iter - warmup_iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs
        
        self.optimizer_mat = torch.optim.Adam((material.get_parameters(self.mat)), lr=self.cfg.OPT.LEARNING_RATE_LGT)
        self.scheduler_mat = torch.optim.lr_scheduler.LambdaLR(self.optimizer_mat, lr_lambda=lambda x: lr_schedule(x, 0.9))

        if self.cfg.OPT.OPTIMIZE_MESH:
            params_mesh = [
                {"params": [self.beta], "lr": self.cfg.OPT.LR_BETA},
                {"params": [self.offsets], "lr": self.cfg.OPT.LR_OFFSETS},
                {"params": self.list_hand_pose, "lr": self.cfg.OPT.LR_HAND_POSE},
                    
            ]
            if self.cfg.DATA.NAME == "kinect" or "syn" in self.cfg.DATA.NAME:
                params_mesh += [
                    {"params": self.list_global_rot, "lr": self.cfg.OPT.LR_GLOBAL_ROT},
                    {"params": self.list_global_transl, "lr": self.cfg.OPT.LR_GLOBAL_TRANSL}
                ]
            elif self.cfg.DATA.NAME == "interhand":
                params_mesh += [
                    {"params": self.list_global_orient, "lr": self.cfg.OPT.LR_GLOBAL_ROT},
                    {"params": self.list_transl, "lr": self.cfg.OPT.LR_GLOBAL_TRANSL}
                ]
            self.optimizer_mesh = torch.optim.SGD(params_mesh, lr = self.cfg.OPT.LR_GEOM)
            self.scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(self.optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9))

        if self.cfg.OPT.OPTIMIZE_LIGHT:
            self.optimizer_light = torch.optim.Adam((self.lgt.parameters()), lr=self.cfg.OPT.LEARNING_RATE_LGT)
            self.scheduler_light = torch.optim.lr_scheduler.LambdaLR(self.optimizer_light, lr_lambda=lambda x: lr_schedule(x, 0.9))
          
    def get_mano_params(self, batch):
        # get batch size of current batch
        N = len(iter(batch.values()).__next__())

        # extract coarse parameters from data
        betas = batch["mano_param"][:, :10]
        transl = batch["mano_param"][:, 10:10+3]
        global_orient = batch["mano_param"][:, 10+3:10+3+3]
        hand_pose = batch["mano_param"][:, 10+3+3:]
        offsets = batch["mano_offsets"]
        
        # add refined offsets
        betas_new = self.beta.expand(N, self.beta.shape[-1]) + betas
        offsets_new = self.offsets.expand(N, self.offsets.shape[0], self.offsets.shape[1]) + offsets
        
        # get data ids corresponding to frames used in this batch
        list_id_data = batch["id_data"]

        global_orient_new = self.list_global_orient[list_id_data] + global_orient
        hand_pose_new = self.list_hand_pose[list_id_data] + hand_pose
        transl_new = self.transl[list_id_data] + transl
        
        return betas_new, global_orient_new, hand_pose_new, transl_new, offsets_new

    def forward_mano(self, batch):
        # get data ids corresponding to frames used in this batch
        list_id_data = batch["id_data"][0]

        # extract coarse parameters from data
        betas = batch["mano_param"][:, :10]
        transl = batch["mano_param"][:, 10:10+3]
        global_orient = batch["mano_param"][:, 10+3:10+3+3]
        hand_pose = batch["mano_param"][:, 10+3+3:]

        # get batch size of current batch
        N = len(iter(batch.values()).__next__())

        # add refined offsets and apply mano
        betas_new = self.beta.expand(N, self.beta.shape[-1]) + betas
        offsets_new = self.offsets.expand(N, self.offsets.shape[0], self.offsets.shape[1])
        list_hand_pose_new = self.list_hand_pose[list_id_data] + hand_pose
        if self.cfg.DATA.NAME == "kinect" or "syn" in self.cfg.DATA.NAME:
            list_global_rot_new = self.list_global_rot[list_id_data] + global_orient
            list_global_transl_new = self.list_global_transl[list_id_data] + transl

            mano_output = self.mano(betas_new, torch.zeros((N, 3), device=self.cfg.DEVICE), list_hand_pose_new, torch.zeros((N, 3), device=self.cfg.DEVICE), offsets_new, flat_hand_mean=True)
            verts = mano_output.vertices

            list_global_rot_mat = so3_exp_map(list_global_rot_new)
            verts = verts @ list_global_rot_mat.transpose(1, 2) + list_global_transl_new
        
        elif self.cfg.DATA.NAME == "interhand":
            list_global_orient_new = self.list_global_orient[list_id_data] + global_orient
            list_transl_new = self.list_transl[list_id_data] + transl

            mano_output = self.mano(betas_new, list_global_orient_new, list_hand_pose_new, list_transl_new, offsets_new)
            verts = mano_output.vertices

        return verts


    def get_mesh(self, verts):
        
        opt_mesh = mesh.Mesh(v_pos=verts[0], base=self.base_mesh)
        with torch.no_grad():
            ou.optix_build_bvh(self.optix_ctx, opt_mesh.v_pos.contiguous(), opt_mesh.t_pos_idx.int(), rebuild=1)
        opt_mesh = mesh.auto_normals(opt_mesh)
        opt_mesh = mesh.compute_tangents(opt_mesh)
        return opt_mesh
    
    def compute_loss(self, buffers, batch, id_step, num_steps, opt_mesh):
        t_iter = (id_step+1) / num_steps
        color_ref = batch['img']

        # Image-space loss, split into a coverage component and a color component
        img_loss  = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:]) 
        img_loss += self.image_loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])

        reg_loss = torch.tensor([0], dtype=torch.float32, device=self.cfg.DEVICE)

        # Monochrome shading regularizer
        reg_loss = reg_loss + regularizer.shading_loss(buffers['diffuse_light'], buffers['specular_light'], color_ref, self.cfg.OPT.LAMBDA_DIFFUSE, self.cfg.OPT.LAMBDA_SPECULAR)

        # Material smoothness regularizer
        reg_loss = reg_loss + regularizer.material_smoothness_grad(buffers['kd_grad'], buffers['ks_grad'], buffers['normal_grad'], lambda_kd=self.cfg.OPT.LAMBDA_KD, lambda_ks=self.cfg.OPT.LAMBDA_KS, lambda_nrm=self.cfg.OPT.LAMBDA_NRM)

        # Chroma regularizer
        reg_loss = reg_loss + regularizer.chroma_loss(buffers['kd'], color_ref, self.cfg.OPT.LAMBDA_CHROMA)

        # Perturbed normal regularizer
        if 'perturbed_nrm_grad' in buffers:
            reg_loss = reg_loss + torch.mean(buffers['perturbed_nrm_grad']) * self.cfg.OPT.LAMBDA_NRM2

        # Laplacian regularizer. 
        if self.cfg.OPT.LAPLACE == "absolute":
            reg_loss = reg_loss + regularizer.laplace_regularizer_const(opt_mesh.v_pos, opt_mesh.t_pos_idx) * self.cfg.OPT.LAPLACE_SCALE * (1 - t_iter)
        elif self.cfg.OPT.LAPLACE == "relative":
            init_mesh = mesh.Mesh(v_pos=batch["mano_verts"][0], base=self.base_mesh)
            reg_loss = reg_loss + regularizer.laplace_regularizer_const(opt_mesh.v_pos - init_mesh.v_pos, init_mesh.t_pos_idx) * self.cfg.OPT.LAPLACE_SCALE * (1 - t_iter)                

        ref = torch.clamp(util.rgb_to_srgb(batch['img'][...,0:3]), 0.0, 1.0)
        opt = torch.clamp(util.rgb_to_srgb(buffers['shaded'][...,0:3]), 0.0, 1.0)
        lpips_loss = self.lpips(opt.permute(0, 3, 1, 2), ref.permute(0, 3, 1, 2), normalize=True)
        reg_loss = reg_loss + self.cfg.OPT.LAMBDA_LPIPS * torch.mean(lpips_loss)

        # Light white balance regularizer
        reg_loss = reg_loss + self.lgt.regularizer() * self.cfg.OPT.W_LGT_REG

        return img_loss, reg_loss
    
    @torch.no_grad()
    def render_buffers(self, batch, opt_mesh, buffers):
        result_dict = {}
        result_dict['ref'] = util.rgb_to_srgb(batch['img'][0, ...,0:3])
        result_dict['ref'] = torch.cat([result_dict['ref'], batch['img'][0, ...,3:4]], dim=2)
        result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][0, ...,0:3])
        result_dict['opt'] = torch.cat([result_dict['opt'], buffers['shaded'][0, ...,3:4]], dim=2)
        
        result_dict['light_image'] = self.lgt.generate_image(self.cfg.IMG_RES)
        result_dict['light_image'] = util.rgb_to_srgb(result_dict['light_image'] / (1 + result_dict['light_image']))
        result_dict['light_image'] = torch.cat([result_dict['light_image'], torch.ones([self.cfg.IMG_RES[0], self.cfg.IMG_RES[1], 1], device=self.cfg.DEVICE)], dim=2)
        
        # white_bg = torch.ones_like(batch['background'])
        result_dict["kd"] = util.rgb_to_srgb(render.render_mesh(self.cfg, self.glctx, opt_mesh, batch["mvp"], batch["campos"], self.lgt, self.cfg.IMG_RES, spp=self.cfg.RENDER.SPP, num_layers=self.cfg.RENDER.LAYERS, msaa=False, bsdf="kd", optix_ctx=self.optix_ctx, denoiser=self.denoiser)['shaded'][..., 0:3])[0]
        result_dict['kd'] = torch.cat([result_dict['kd'], buffers['shaded'][0, ...,3:4]], dim=2)

        result_dict["ks"] = render.render_mesh(self.cfg, self.glctx, opt_mesh, batch["mvp"], batch["campos"], self.lgt, self.cfg.IMG_RES, spp=self.cfg.RENDER.SPP, num_layers=self.cfg.RENDER.LAYERS, msaa=False, bsdf="ks", optix_ctx=self.optix_ctx, denoiser=self.denoiser)['shaded'][0, ..., 0:3]
        result_dict['ks'] = torch.cat([result_dict['ks'], buffers['shaded'][0, ...,3:4]], dim=2)
        
        result_dict["normal"] = render.render_mesh(self.cfg, self.glctx, opt_mesh, batch["mvp"], batch["campos"], self.lgt, self.cfg.IMG_RES, spp=self.cfg.RENDER.SPP, num_layers=self.cfg.RENDER.LAYERS, msaa=False, bsdf="normal", optix_ctx=self.optix_ctx, denoiser=self.denoiser)['shaded'][0, ..., 0:3]
        result_dict['normal'] = torch.cat([result_dict['normal'], buffers['shaded'][0, ...,3:4]], dim=2)

        result_image = torch.cat([result_dict['ref'], result_dict['opt'], result_dict['light_image'], result_dict["kd"], result_dict["ks"], result_dict["normal"]], axis=1)
        if not self.cfg.MAT.NO_PERTURBED_NRM:
            result_dict["perturbed_nrm"] = (buffers['perturbed_nrm'][0, ...,0:3] + 1.0) * 0.5
            result_dict['perturbed_nrm'] = torch.cat([result_dict['perturbed_nrm'], buffers['shaded'][0, ...,3:4]], dim=2)
            result_image = torch.cat([result_image, result_dict["perturbed_nrm"]], axis=1)
        
        result_dict["diffuse_light"] = util.rgb_to_srgb(buffers['diffuse_light'][..., 0:3])[0]
        result_dict['diffuse_light'] = torch.cat([result_dict['diffuse_light'], buffers['shaded'][0, ...,3:4]], dim=2)
        result_dict["specular_light"] = util.rgb_to_srgb(buffers['specular_light'][..., 0:3])[0]
        result_dict['specular_light'] = torch.cat([result_dict['specular_light'], buffers['shaded'][0, ...,3:4]], dim=2)

        result_image = torch.cat([result_image, result_dict["diffuse_light"], result_dict["specular_light"]], axis=1)

        return result_image, result_dict
    
    def forward(self, batch, train=False):
        if train:
            if self.cfg.OPT.OPTIMIZE_LIGHT:
                self.lgt.update_pdf()
            verts = self.forward_mano(batch)
            opt_mesh = self.get_mesh(verts)
            buffers = render.render_mesh(self.cfg, self.glctx, opt_mesh, batch["mvp"], batch["campos"], self.lgt, self.cfg.IMG_RES, spp=self.cfg.RENDER.SPP, num_layers=self.cfg.RENDER.LAYERS, msaa=True, background=batch['background'], optix_ctx=self.optix_ctx, denoiser=self.denoiser)
        else:
            with torch.no_grad(): 
                verts = self.forward_mano(batch)
                opt_mesh = self.get_mesh(verts)
                buffers = render.render_mesh(self.cfg, self.glctx, opt_mesh, batch["mvp"], batch["campos"], self.lgt, self.cfg.IMG_RES, spp=self.cfg.RENDER.SPP, num_layers=self.cfg.RENDER.LAYERS, msaa=False, background=batch['background'], optix_ctx=self.optix_ctx, denoiser=self.denoiser)
            
        return opt_mesh, buffers

    def optimize_epoch(self, dataloader_train, id_epoch, out_train_log_dir):
        num_steps = self.cfg.OPT.EPOCHS * len(dataloader_train)
        running_img_loss = 0.0
        running_reg_loss = 0.0
        running_total_loss = 0.0
        
        for id_batch, batch in enumerate(dataloader_train):
            id_step = id_epoch * len(dataloader_train) + id_batch
            
            self.optimizer_mat.zero_grad()
            if self.cfg.OPT.OPTIMIZE_MESH:
                self.optimizer_mesh.zero_grad()
            if self.cfg.OPT.OPTIMIZE_LIGHT:
                self.optimizer_light.zero_grad()
            
            # ==============================================================================================
            #  Initialize training
            # ==============================================================================================

            batch = mix_background(batch)
                
            opt_mesh, buffers = self.forward(batch, train=True)
            img_loss, reg_loss = self.compute_loss(buffers, batch, id_step, num_steps, opt_mesh)
            total_loss = img_loss + reg_loss

            # ==============================================================================================
            #  Backpropagate
            # ==============================================================================================
            total_loss.backward()

            if self.cfg.OPT.OPTIMIZE_LIGHT:
                self.lgt.base.grad *= 64

            self.optimizer_mat.step()
            self.scheduler_mat.step()
            if self.cfg.OPT.OPTIMIZE_MESH:
                self.optimizer_mesh.step()
                self.scheduler_mesh.step()
            if self.cfg.OPT.OPTIMIZE_LIGHT:
                self.optimizer_light.step()
                self.scheduler_light.step()

            # ==============================================================================================
            #  Clamp trainables to reasonable range
            # ==============================================================================================
            with torch.no_grad():
                if 'kd' in self.mat:
                    self.mat['kd'].clamp_()
                if 'ks' in self.mat:
                    self.mat['ks'].clamp_()
                if 'normal' in self.mat:
                    self.mat['normal'].clamp_()
                    self.mat['normal'].normalize_()
                if self.lgt is not None:
                    self.lgt.clamp_(min=0.01) # For some reason gradient dissapears if light becomes 0

            # ==============================================================================================
            #  Log & save outputs
            # ==============================================================================================
            
            running_img_loss += img_loss.item()
            running_reg_loss += reg_loss.item()
            running_total_loss += total_loss.item()

            # log
            if id_step % self.cfg.LOG.SAVE_INTERVAL == (self.cfg.LOG.SAVE_INTERVAL-1):
                opt_mesh, buffers = self.forward(batch, train=False)
                result_image, result_dict = self.render_buffers(batch, opt_mesh, buffers)
                util.save_image(f"{out_train_log_dir}/{id_step:05d}.png", result_image.detach().cpu().numpy())

                logger.info(f"[{id_step:>5d} / {num_steps:>5d}]  img_loss: {running_img_loss / self.cfg.LOG.SAVE_INTERVAL:>7f}  reg_loss: {running_reg_loss / self.cfg.LOG.SAVE_INTERVAL:>7f} total_loss: {running_total_loss / self.cfg.LOG.SAVE_INTERVAL:>7f}")
                running_img_loss = 0.0
                running_reg_loss = 0.0
                running_total_loss = 0.0

            

    def test_epoch(self, dataloader_test, id_epoch, out_test_log_dir):
        
        mse_values = []
        psnr_values = []
        ssim_values = []
        msssim_values = []
        lpips_values = []

        # Hack validation to use high sample count and no denoiser
        _n_samples = self.cfg.RENDER.N_SAMPLES
        _denoiser = self.denoiser
        self.cfg.RENDER.N_SAMPLES = 32
        self.denoiser = None
        
        out_val_dir = f"{out_test_log_dir}/epoch_{id_epoch:02d}"; utils.create_dir(out_val_dir, True)
        logger.info("Running evaluation on test")
        with open(f"{out_val_dir}/metrics.txt", 'a') as fout:
            fout.write('ID, MSE, PSNR, SSIM, MSSIM, LPIPS\n')
            # fout.write(f"Epoch: {id_epoch}\n")
            for id_batch, batch in enumerate(tqdm(dataloader_test)):
                batch = mix_background(batch)
                opt_mesh, buffers = self.forward(batch, train=False)
                result_image, result_dict = self.render_buffers(batch, opt_mesh, buffers)

                # Compute metrics
                opt = torch.clamp(result_dict['opt'], 0.0, 1.0)[..., :3]
                ref = torch.clamp(result_dict['ref'], 0.0, 1.0)[..., :3]

                mse = torch.nn.functional.mse_loss(opt, ref, size_average=None, reduce=None, reduction='mean').item()
                mse_values.append(float(mse))
                psnr = util.mse_to_psnr(mse)
                psnr_values.append(float(psnr))
                ssim = calculate_ssim(opt.permute(2, 0, 1)[None, :, :, :], ref.permute(2, 0, 1)[None, :, :, :])
                ssim_values.append(float(ssim))
                msssim = calculate_msssim(opt.permute(2, 0, 1)[None, :, :, :], ref.permute(2, 0, 1)[None, :, :, :])
                msssim_values.append(float(msssim))
                lpips_value = self.lpips(opt.permute(2, 0, 1)[None, :, :, :], ref.permute(2, 0, 1)[None, :, :, :], normalize=True).item()
                lpips_values.append(float(lpips_value))

                line = f"{id_batch:>5d} {mse:>7f} {psnr:>7f} {ssim:>7f} {msssim:>7f} {lpips_value:>7f}\n"
                fout.write(str(line))
                util.save_image(f"{out_val_dir}/{id_batch:05d}.png", result_image.detach().cpu().numpy())

            avg_mse = np.mean(np.array(mse_values))
            avg_psnr = np.mean(np.array(psnr_values))
            avg_ssim = np.mean(np.array(ssim_values))
            avg_msssim = np.mean(np.array(msssim_values))
            avg_lpips = np.mean(np.array(lpips_values))
            line = f"Average\n{avg_mse:04f}, {avg_psnr:04f}, {avg_ssim:04f}, {avg_msssim:04f}, {avg_lpips:04f}\n"
            fout.write(str(line))
            logger.info("MSE,      PSNR,       SSIM,      MSSIM,     LPIPS")
            logger.info(line[8:])
        
        # Restore sample count and denoiser
        self.cfg.RENDER.N_SAMPLES = _n_samples
        self.denoiser = _denoiser

    def optimize(self, log_dir):
        logger.info(f"Data size: {len(self.dataset)}")
        dataloader_train = torch.utils.data.DataLoader(self.dataset, batch_size=self.cfg.BATCH_SIZE, collate_fn=self.dataset.collate, shuffle=False)
        dataloader_test = torch.utils.data.DataLoader(self.dataset, batch_size=self.cfg.BATCH_SIZE, collate_fn=self.dataset.collate, shuffle=False)
        
        out_train_log_dir = f"{log_dir}/train"; utils.create_dir(out_train_log_dir, True)
        out_test_log_dir = f"{log_dir}/test"; utils.create_dir(out_test_log_dir, True)
        
        for id_epoch in range(self.cfg.OPT.EPOCHS):
            logger.info(f"------------- Epoch {id_epoch} -------------")
            self.optimize_epoch(dataloader_train, id_epoch, out_train_log_dir)
            self.test_epoch(dataloader_test, id_epoch, out_test_log_dir)
        logger.info(f"Optimization done!")


    def save_optimized_mesh_and_light(self, out_opt_dir):
        data = self.dataset[0]
        beta_data = data["mano_param"][0, :10]
        beta_new = self.beta + beta_data
        
        mano_output = self.mano(beta_new[None, :], torch.zeros((1, 3), device=self.cfg.DEVICE), torch.zeros((1, 15*3), device=self.cfg.DEVICE), torch.zeros((1, 3), device=self.cfg.DEVICE), self.offsets[None, :, :], flat_hand_mean=self.cfg.DATA.NAME=="kinect")
        verts = mano_output.vertices
        final_mesh = self.get_mesh(verts)
        obj.write_obj(out_opt_dir, final_mesh)
        light.save_env_map(f"{out_opt_dir}/probe.hdr", self.lgt)

        np.save(f"{out_opt_dir}/beta.npy", beta_new.detach().cpu().numpy())
        np.save(f"{out_opt_dir}/offsets.npy", self.offsets.detach().cpu().numpy())
