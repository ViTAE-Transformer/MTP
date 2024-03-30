import os
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm

address = '/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/loveda/rvsa-b-upernet-512-mae-mtp-loveda_reuse_decoder/predict/submit/'
new_address = '/diwang22/work_dir/multitask_pretrain/finetune/Semantic_Segmentation/loveda/rvsa-b-upernet-512-mae-mtp-loveda_reuse_decoder/predict/new_submit/'

os.makedirs(new_address, exist_ok=True)

files = glob(os.path.join(address,'*.png'))

for i in tqdm(range(len(files))):
    _, basename = os.path.split(files[i])
    img = np.array(Image.open(files[i]))
    img -= 1
    filename = os.path.join(new_address, basename)
    img = Image.fromarray(img.astype('uint8'))
    img.save(filename)




