# first set up which gpu to use
import os
gpu_ids = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids}"

# import libraries
import numpy as np
from termcolor import colored, cprint

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True


from datasets.dataloader import CreateDataLoader, get_data_generator
from models.base_model import create_model
from utils.util_3d import render_sdf, render_mesh, sdf_to_mesh, save_mesh_as_gif
from utils.demo_util import SDFusionOpt
from pytorch3d.io import save_obj

seed = 2023
opt = SDFusionOpt(gpu_ids=gpu_ids, seed=seed)
device = opt.device

# initialize SDFusion model
ckpt_path = 'saved_ckpt/sdfusion-snet-all.pth'
dset="snet"
opt.init_model_args(ckpt_path=ckpt_path)
opt.init_dset_args(dataset_mode=dset)
SDFusion = create_model(opt)
cprint(f'[*] "{SDFusion.name()}" loaded.', 'cyan')

# unconditional generation
out_dir = 'demo_results'
if not os.path.exists(out_dir): os.makedirs(out_dir)

ngen = 6
ddim_steps = 100
ddim_eta = 0.

sdf_gen = SDFusion.uncond(ngen=ngen, ddim_steps=ddim_steps, ddim_eta=ddim_eta)
print('sdf shape', sdf_gen.shape)
mesh_gen = sdf_to_mesh(sdf_gen)

verts, faces = mesh_gen.get_mesh_verts_faces(0)

save_obj('demo_results/output.obj', verts, faces)


# vis as gif
gen_name = f'{out_dir}/uncond.gif'
save_mesh_as_gif(SDFusion.renderer, mesh_gen, nrow=3, out_name=gen_name)
