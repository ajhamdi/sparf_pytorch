import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
import pandas as pd
import json
from torchvision import transforms

def get_image_to_tensor_balanced(image_size=0):
    ops = []
    if image_size > 0:
        ops.append(transforms.Resize(image_size))
    ops.extend(
        [transforms.ToTensor(), transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]
    )
    return transforms.Compose(ops)


def get_mask_to_tensor():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]
    )

class SPARFDataset(torch.utils.data.Dataset):
    """SPARF: posed Multi-view image dataset , Hamdi et.al,  2022
    A class for loading multi-view posed images of SPARF dataset.

    Parameters:
    -----------
    data_dir : str
        The path to the directory containing the dataset.
    views_split : str, optional
        The split of views to be loaded. Can be "train", "test", or "hard".
        Default is "train".
    object_class : str, optional
        The category of objects to be loaded. Default is "car". possible classes: ["watercraft", "rifle", "display", "lamp", "speaker", "cabinet", "chair", "bench", "car", "airplane", "sofa", "table", "phone"]
    n_views : int or None, optional
        The number of views to be loaded. If None, all available views are loaded.
        Default is None.
    dset_partition : int, optional
        The partition of the dataset to be loaded. Can be -1 to include all 20 partitions
        of the data of that class [0:19], otherwise takes just a portion of the data.
        Default is -1.
    z_near : float, optional
        The minimum depth value of the camera. Default is 0.01.
    z_far : float, optional
        The maximum depth value of the camera. Default is 10.0.
    return_as_lists : bool, optional
        Whether to return the data as a dictionary of lists (each list has length `n_views`)
        or as a dictionary of tensors. Default is False.

    Examples:
    ---------
    # Initialize the dataset object
    data_dir = "/path/to/data"
    dset = SPARFDataset(data_dir)

    # Get the first data sample
    data = dset[0]


    # Get the images and masks of the first data sample
    images = data["images"]
    masks = data["masks"]
    ipyplot.plot_images(images,)
    """

    def __init__(self, path, views_split="train", z_near=0.01, z_far=1000.0, n_views=None, object_class="car", dset_partition=-1,return_as_lists=False):
        super().__init__()
        self.views_split = views_split
        self.data_splits = ["train","test"]
        self.return_as_lists = return_as_lists

        self.object_class = object_class
        self.dset_partition = dset_partition
        self.object_class_dir = os.path.join(os.getcwd(),path, self.object_class)

        splits = pd.read_csv(os.path.join(os.getcwd(),path, "SNRL_splits.csv"), sep=",", dtype=str)
        avail_files = sorted(list(os.listdir(self.object_class_dir)))
        if self.dset_partition == -1:
            splits = splits[splits.file.isin(avail_files) & splits.classlabel.isin([
                str(self.object_class)])]

        else:
            splits = splits[splits.file.isin(avail_files) & splits.partition.isin([str(x) for x in range(
                self.dset_partition+1)]) & splits.classlabel.isin([str(self.object_class)])]

        self.model_ids = list(splits[splits.split.isin(self.data_splits)]["file"])
        # print(len(self.model_ids))
        self.synset_ids = [
            self.object_class for _ in range(len(self.model_ids))]

        # path = os.path.join(path, views_split)
        self.base_path = self.object_class_dir
        print("Loading NeRF synthetic dataset", self.base_path)
        # trans_files = []
        # TRANS_FILE = "transforms_{}.json".format(self.views_split)
        # for root, directories, filenames in os.walk(self.base_path):
        #     if TRANS_FILE in filenames:
        #         trans_files.append(os.path.join(root, TRANS_FILE))
        trans_files = [os.path.join(self.base_path, c_id, "transforms_{}.json".format(self.views_split))for c_id in self.model_ids]
        self.trans_files = trans_files
        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()

        self.z_near = z_near
        self.z_far = z_far
        self.lindisp = False
        self.n_views = n_views

        print("{} instances in split {}".format(len(self.trans_files), views_split))

    def __len__(self):
        return len(self.trans_files)

    def _check_valid(self, index):
        if self.n_views is None:
            return True
        trans_file = self.trans_files[index]
        dir_path = os.path.dirname(trans_file)
        try:
            with open(trans_file, "r") as f:
                transform = json.load(f)
        except Exception as e:
            print("Problematic transforms.json file", trans_file)
            print("JSON loading exception", e)
            return False
        if len(transform["frames"]) < self.n_views:
            print("requested number of views ({}) is more than available {} views".format(self.n_views,len(transform["frames"])))
            return False
        # if len(glob.glob(os.path.join(dir_path, "*.png"))) != self.n_views:
        #     return False
        return True

    def __getitem__(self, index):
        if not self._check_valid(index):
            return {}

        trans_file = self.trans_files[index]
        dir_path = os.path.dirname(trans_file)
        with open(trans_file, "r") as f:
            transform = json.load(f)

        imgs = []
        bboxes = []
        masks = []
        poses = []
        for frame in transform["frames"]:
            fpath = frame["file_path"]
            basename = os.path.splitext(os.path.basename(fpath))[0]
            obj_path = os.path.join(dir_path,self.views_split, "{}.png".format(basename))
            img = imageio.imread(obj_path)
            mask = self.mask_to_tensor(img[..., 3])
            rows = np.any(img[..., 3], axis=1)
            cols = np.any(img[..., 3], axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            if len(rnz) == 0:
                cmin = rmin = 0
                cmax = mask.shape[-1]
                rmax = mask.shape[-2]
            else:
                rmin, rmax = rnz[[0, -1]]
                cmin, cmax = cnz[[0, -1]]
            bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)

            img_tensor = self.image_to_tensor(img[..., :3])
            img = img_tensor * mask + (
                1.0 - mask
            )  # solid white background where transparent
            imgs.append(img)
            bboxes.append(bbox)
            masks.append(mask)
            poses.append(torch.tensor(frame["transform_matrix"]))
        if not self.return_as_lists:
            imgs = torch.stack(imgs)
            masks = torch.stack(masks)
            bboxes = torch.stack(bboxes)
            poses = torch.stack(poses)

        H, W = imgs[0].shape[-2:]
        camera_angle_x = transform.get("camera_angle_x")
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
        # print(bboxes.mean(), masks.mean())
        result = {
            "path": dir_path,
            "img_id": index,
            "focal": focal,
            "images": imgs[:self.n_views],
            "masks": masks[:self.n_views],
            "bbox": bboxes[:self.n_views],
            "poses": poses[:self.n_views],
        }
        return result

