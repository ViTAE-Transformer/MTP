import os
import torch

ckpt_paths = ['/diwang22/work_dir/multitask_pretrain/pretrain/avg/with_background/vit_b_rvsa_224_mae_samrs_mtp_three/last_vit_b_rvsa_ss_is_rd_pretrn_model.pth']

for ckpt_path in ckpt_paths:
    path, filename = os.path.split(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']

    model_dict = {}

    for k, v in ckpt.items():
        if 'encoder' in k:
            k = k.replace('encoder', 'backbone')
        elif 'rotdetdecoder' in k:
            k = k.replace('rotdetdecoder.', '')

        model_dict[k] = v

    filename = os.path.join(path, filename[:-4] + '_rot' +'.pth')
    torch.save({'state_dict': model_dict}, filename)
