from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import torch
import os
import tqdm
import glob 
import argparse
import json
from skimage.segmentation import find_boundaries
import numpy as np

WHITE_PIXEL = torch.tensor([253, 231, 36, 255])
BLACK_PIXEL = torch.tensor([68, 1, 84, 255])
MAX_WIDTH, MAX_HEIGHT = 1728, 960

def pad_sequence(img_width, img_height, pad_width, pad_height):
    horizontal, vertical = pad_width-img_width, pad_height-img_height
    
    l = horizontal // 2
    r = l + horizontal % 2
    
    u = vertical // 2
    d = u + vertical % 2
    
    pad_sequence = [l, u, r, d]
    return pad_sequence

# Read xray image as rgb
# Return tensor(3 x w x h): RGB image tensor
def read_image(img_path, greyscale=False):
    image = Image.open(img_path)
    
    if greyscale:
        image = ImageOps.grayscale(image)
    
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    img_tensor = transform(image)
    
    return img_tensor


# Read label image and convert to greyscale
# Return tensor(w x h): Greyscale image tensor
def read_label(img_path):
    img_tensor = read_image(img_path)
    
    label_tensor = torch.all(img_tensor.permute(1,2,0) == 255, dim=-1).unsqueeze(0)

    return label_tensor
def calculate_bboxes(imgs: torch.Tensor, loosen_amount: float = 0.03, region = None):
        """
        Return the bounding boxes of tensor
        - imgs: input tensor (b x h x w)
        Return:
        - bboxes: List[Tuple(float, float, float, float)] - List of bounding boxes (x_center y_center w h) for all batches in the tensor
        """
        bboxes = []
        for t in imgs:
            index = (t == 1).nonzero()
            
            rmax_id = torch.argmax(index[:,0])
            cmax_id = torch.argmax(index[:,1])
            rmin_id = torch.argmin(index[:,0])
            cmin_id = torch.argmin(index[:,1])
            
            r0, c0 = index[rmin_id,0], index[cmin_id, 1]
            r1, c1 = index[rmax_id,0], index[cmax_id, 1]
            h, w = index[rmax_id,0]-index[rmin_id,0], index[cmax_id, 1]-index[cmin_id, 1]
            r_center, c_center = r0+h/2, c0+w/2
            # loose_region = (0,0)
            if region == "NT":
                loose_region = (r1-r0, c1-c0)
            # check if region is a tensor
            elif isinstance(region, torch.Tensor) and region.shape == torch.Size([2]):
                loose_region = region*torch.tensor(t.shape)

            else:
                loose_region = t.shape
            # print(loosen_amount*loose_region[0]/t.shape[0])
            # print(r0, c0, r1, c1, w, h, r_center, c_center)
            r0 = max(r0 / t.shape[0] - loosen_amount*loose_region[0]/t.shape[0], torch.tensor(0))
            r1 = min(r1 / t.shape[0] + loosen_amount*loose_region[0]/t.shape[0], torch.tensor(1))
            c0 = max(c0 / t.shape[1] - loosen_amount*loose_region[1]/t.shape[1], torch.tensor(0))
            c1 = min(c1 / t.shape[1] + loosen_amount*loose_region[1]/t.shape[1], torch.tensor(1))

            
            r_center, c_center = r_center/t.shape[0], c_center/t.shape[1]
            h, w = r1-r0, c1-c0
            # bboxes.append(torch.stack((r0, c0, r1, c1)))
            bboxes.append(torch.stack((c_center, r_center, w, h)))
        return bboxes
def get_boundary(imgs: torch.Tensor):
    '''
    Return the boundary of the image
    - imgs: input tensor (b x h x w)
    Return:
    - boundary: (n x y x x) tensor - boundary of the image
    '''
    boundary_tensor = torch.tensor(find_boundaries(imgs[0]).astype(np.uint8))
    boundary = torch.div(torch.nonzero(boundary_tensor), torch.tensor(boundary_tensor[0].size()))
    return boundary
def convert(path, loosen_amount, segment, loose_region, combine_head = False):
    task = path.split("/")[-1]
    # print(task)
    in_paths = glob.glob(os.path.join(path, "label/*"))
    img_paths = glob.glob(os.path.join(path, "images/*"))
    # print(img_paths)
    # export img_paths to .txt file
    with open(path+'.txt', "w") as f:
        for img_path in img_paths:
            f.write(img_path + "\n")
        print("Exported image paths to", path+'.txt')

    if not os.path.exists(os.path.join(path, "labels")):
        os.mkdir(os.path.join(path, "labels"))
    for in_path in tqdm.tqdm(in_paths, desc=f"Converting {task} labels"):
        label_tensor = read_label(in_path)
        if combine_head or loose_region == "head":
            head_label_tensor = read_label(in_path.replace("label", "head_label_binary"))
            # label_tensor = torch.logical_or(label_tensor, head_label_tensor)
        out_path = os.path.join(path, "labels", os.path.basename(in_path).replace(".png", ".txt"))
        if not segment:
            head_bboxes=[]
            bboxes = []
            if combine_head or loose_region == "head":
                head_bboxes = calculate_bboxes(head_label_tensor, 0, None)
                if not combine_head:
                    # get last two elements of head_bboxes
                    bboxes = calculate_bboxes(label_tensor, loosen_amount, head_bboxes[0][-2:])
                else:
                    label_tensor = torch.logical_or(label_tensor, head_label_tensor)
                    if loose_region == "head":
                        bboxes = calculate_bboxes(label_tensor, loosen_amount, head_bboxes[0][-2:])
                    else:
                        bboxes = calculate_bboxes(label_tensor, loosen_amount, loose_region)
            else:
                bboxes = calculate_bboxes(label_tensor, loosen_amount, loose_region)
            with open(out_path, "w") as f:
                for bbox in bboxes:
                    f.write("0 ")
                    f.write(" ".join([str(x.item()) for x in bbox]))
                    f.write("\n")
                for bbox in head_bboxes:
                    f.write("1 ")
                    f.write(" ".join([str(x.item()) for x in bbox]))
                    f.write("\n")
            
        else:
            boundary = get_boundary(label_tensor)
            # print(boundary)
            with open(out_path, "w") as f:
                f.write("0 ")
                for y,x in boundary:
                    f.write(" ".join([str(x.item()), str(y.item())]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default = 'cfg/custom/config.json',help="Path to config file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
        for path in config["data_paths"]:
            convert(config["data_paths"][path], config["loosen_amount"], config["segment"], config["loose_region"], config["combine_head"])
    # convert('../data/baby_small/train')
if __name__== '__main__':
    main()