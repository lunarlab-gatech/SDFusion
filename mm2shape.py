# first set up which gpu to use
import os
gpu_ids = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids}"

# import libraries
from termcolor import colored, cprint

from pytorch3d.io import save_obj
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torchvision.utils as vutils

from models.base_model import create_model
from utils.util_3d import read_sdf, render_sdf, render_mesh, sdf_to_mesh, save_mesh_as_gif

# options for the model. please check `utils/demo_util.py` for more details
from utils.demo_util import SDFusionMM2ShapeOpt

seed = 2023
opt = SDFusionMM2ShapeOpt(gpu_ids=gpu_ids, seed=seed)
device = opt.device

# initialize SDFusion model
ckpt_path = 'saved_ckpt/sdfusion-mm2shape.pth'
opt.init_model_args(ckpt_path=ckpt_path)

SDFusion = create_model(opt)
cprint(f'[*] "{SDFusion.name()}" loaded.', 'cyan')

from utils.demo_util import preprocess_image, get_shape_mask, tensor_to_pil
import torchvision.transforms as transforms
# mm2shape
out_dir = 'demo_results'
if not os.path.exists(out_dir): os.makedirs(out_dir)

# load input shape
sdf_path = 'demo_data/chair-IKEA-FUSION.h5'
sdf = read_sdf(sdf_path)
sdf = sdf.clamp(-.2, .2)

# get partial shape
mask_mode = 'top' # what to keep. check: demo_util.get_shape_mask for other options

# get text
input_txt = "chair with one leg"

# get image
input_img = "demo_data/revolving-chair.jpg"
input_mask = "demo_data/revolving-chair-mask.png"

img_for_vis, img_clean = preprocess_image(input_img, input_mask)
mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.Resize((256, 256)),
])
img_clean = transforms(img_clean).unsqueeze(0)

# pack into a dict
test_data = {
    'sdf': sdf,
    'img': img_clean,
    'text': [input_txt],
}

rend_sdf = render_sdf(SDFusion.renderer, sdf.to(device))


# inference

# ddim_steps = 100
ddim_steps = 50
ddim_eta = 0.1
uc_scale = 5.

# demo: run four sets of scales
txt_img_scales = [(0., 0.), (1., 0.), (0., 1.), (1., 1.)]


for txt_scale, img_scale in txt_img_scales:
    SDFusion.mm_inference(test_data, mask_mode=mask_mode, ddim_steps=ddim_steps, ddim_eta=ddim_eta, uc_scale=uc_scale,
    txt_scale=txt_scale, img_scale=img_scale)
    # save the generation results
    sdf_gen = SDFusion.gen_df
    mesh_gen = sdf_to_mesh(sdf_gen)
    gen_name = f'{out_dir}/mm2shape-gen_shape-txt_{txt_scale}-img_{img_scale}.gif'
    save_mesh_as_gif(SDFusion.renderer, mesh_gen, nrow=3, out_name=gen_name)

# save input partial shape as well
sdf_part = SDFusion.x_part
mesh_part = sdf_to_mesh(sdf_part)
part_name = f'{out_dir}/mm2shape-part_shape.gif'
save_mesh_as_gif(SDFusion.renderer, mesh_part, nrow=3, out_name=part_name)
