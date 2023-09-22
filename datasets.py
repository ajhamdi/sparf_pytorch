import json
import os 
import pandas as pd
import warnings
from extra_utils import sort_jointly , Normalize
import svox2
import trimesh
import torch
import numpy as np 
import torchvision.transforms as transforms

from Svox2.opt.reflect import  SparseRadianceFields 

class ShapeNetBase(torch.utils.data.Dataset):
    """
    'ShapeNetBase' implements a base Dataset for ShapeNet and R2N2 with helper methods.
    It is not intended to be used on its own as a Dataset for a Dataloader. Both __init__
    and __getitem__ need to be implemented.
    """

    def __init__(self):
        """
        Set up lists of synset_ids and model_ids.
        """
        self.synset_ids = []
        self.model_ids = []
        self.synset_inv = {}
        self.synset_start_idxs = {}
        self.synset_num_models = {}
        self.shapenet_dir = ""
        self.model_dir = "model.obj"
        self.load_textures = True
        self.texture_resolution = 4

    def __len__(self):
        """
        Return number of total models in the loaded dataset.
        """
        return len(self.model_ids)

    def __getitem__(self, idx) :
        """
        Read a model by the given index. Need to be implemented for every child class
        of ShapeNetBase.
        Args:
            idx: The idx of the model to be retrieved in the dataset.
        Returns:
            dictionary containing information about the model.
        """
        raise NotImplementedError(
            "__getitem__ should be implemented in the child class of ShapeNetBase"
        )

    def _get_item_ids(self, idx) :
        """
        Read a model by the given index.
        Args:
            idx: The idx of the model to be retrieved in the dataset.
        Returns:
            dictionary with following keys:
            - synset_id (str): synset id
            - model_id (str): model id
        """
        model = {}
        model["synset_id"] = self.synset_ids[idx]
        model["model_id"] = self.model_ids[idx]
        return model



class ShapeNetCore(ShapeNetBase):
    """
    This class loads ShapeNetCore from a given directory into a Dataset object.
    ShapeNetCore is a subset of the ShapeNet dataset and can be downloaded from
    https://www.shapenet.org/.
    """

    def __init__(
        self,
        data_dir,
        split,
        nb_points,
        synsets=None,
        version: int = 2,
        load_textures: bool = False,
        texture_resolution: int = 4,
        dset_norm: str = "inf",
        simplified_mesh=False
    ):
        """
        Store each object's synset id and models id from data_dir.
        Args:
            data_dir: Path to ShapeNetCore data.
            synsets: List of synset categories to load from ShapeNetCore in the form of
                synset offsets or labels. A combination of both is also accepted.
                When no category is specified, all categories in data_dir are loaded.
            version: (int) version of ShapeNetCore data in data_dir, 1 or 2.
                Default is set to be 1. Version 1 has 57 categories and verions 2 has 55
                categories.
                Note: version 1 has two categories 02858304(boat) and 02992529(cellphone)
                that are hyponyms of categories 04530566(watercraft) and 04401088(telephone)
                respectively. You can combine the categories manually if needed.
                Version 2 doesn't have 02858304(boat) or 02834778(bicycle) compared to
                version 1.
            load_textures: Boolean indicating whether textures should loaded for the model.
                Textures will be of type TexturesAtlas i.e. a texture map per face.
            texture_resolution: Int specifying the resolution of the texture map per face
                created using the textures in the obj file. A
                (texture_resolution, texture_resolution, 3) map is created per face.
        """
        super().__init__()
        self.shapenet_dir = data_dir
        self.nb_points = nb_points
        self.load_textures = load_textures
        self.texture_resolution = texture_resolution
        self.dset_norm = dset_norm
        self.split = split
        self.simplified_mesh = simplified_mesh

        if version not in [1, 2]:
            raise ValueError("Version number must be either 1 or 2.")
        self.model_dir = "model.obj" if version == 1 else "models/model_normalized.obj"
        # if self.simplified_mesh:
        #     self.model_dir = "models/model_normalized_SMPLER.obj"
        splits = pd.read_csv(os.path.join(
            self.shapenet_dir, "shapenet_split.csv"), sep=",", dtype=str)

        dict_file = "shapenet_synset_dict_v%d.json" % version
        with open(os.path.join(self.shapenet_dir, dict_file), "r") as read_dict:
            self.synset_dict = json.load(read_dict)

        self.synset_inv = {label: offset for offset,
                           label in self.synset_dict.items()}

        if synsets is not None:

            synset_set = set()
            for synset in synsets:
                if (synset in self.synset_dict.keys()) and (
                    os.path.isdir(os.path.join(data_dir, synset))
                ):
                    synset_set.add(synset)
                elif (synset in self.synset_inv.keys()) and (
                    (os.path.isdir(os.path.join(data_dir, self.synset_inv[synset])))
                ):
                    synset_set.add(self.synset_inv[synset])
                else:
                    msg = (
                        "Synset category %s either not part of ShapeNetCore dataset "
                        "or cannot be found in %s."
                    ) % (synset, data_dir)
                    warnings.warn(msg)

        else:
            synset_set = {
                synset
                for synset in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, synset))
                and synset in self.synset_dict
            }

        synset_not_present = set(
            self.synset_dict.keys()).difference(synset_set)
        [self.synset_inv.pop(self.synset_dict[synset])
         for synset in synset_not_present]

        if len(synset_not_present) > 0:
            msg = (
                "The following categories are included in ShapeNetCore ver.%d's "
                "official mapping but not found in the dataset location %s: %s"
                ""
            ) % (version, data_dir, ", ".join(synset_not_present))
            warnings.warn(msg)

        for synset in synset_set:
            self.synset_start_idxs[synset] = len(self.synset_ids)
            for model in os.listdir(os.path.join(data_dir, synset)):
                if not os.path.exists(os.path.join(data_dir, synset, model, self.model_dir)):
                    msg = (
                        "Object file not found in the model directory %s "
                        "under synset directory %s."
                    ) % (model, synset)

                    continue
                self.synset_ids.append(synset)
                self.model_ids.append(model)
            model_count = len(self.synset_ids) - self.synset_start_idxs[synset]
            self.synset_num_models[synset] = model_count
        self.model_ids, self.synset_ids = sort_jointly(
            [self.model_ids, self.synset_ids], dim=0)
        self.classes = sorted(list(self.synset_inv.keys()))
        self.label_by_number = {k: v for v, k in enumerate(self.classes)}

        split_model_ids, split_synset_ids = [], []
        for ii, model in enumerate(self.model_ids):
            found = splits[splits.modelId.isin([model])]["split"]
            if len(found) > 0:
                if found.item() in self.split:
                    split_model_ids.append(model)
                    split_synset_ids.append(self.synset_ids[ii])
        self.model_ids = split_model_ids
        self.synset_ids = split_synset_ids

    def __getitem__(self, idx: int):
        """
        Read a model by the given index.
        Args:
            idx: The idx of the model to be retrieved in the dataset.
        Returns:
            dictionary with following keys:
            - verts: FloatTensor of shape (V, 3).
            - faces: LongTensor of shape (F, 3) which indexes into the verts tensor.
            - synset_id (str): synset id
            - model_id (str): model id
            - label (str): synset label.
        """
        model = self._get_item_ids(idx)
        model_path = os.path.join(
            self.shapenet_dir, model["synset_id"], model["model_id"], self.model_dir
        )
        print(model_path)
        mesh = trimesh.load(model_path,force="mesh")
        # mesh = mesh.apply_scale(1.0)
        smapled_points = mesh.sample(self.nb_points)

        label_str = self.synset_dict[model["synset_id"]]
        return self.label_by_number[label_str], mesh, smapled_points

class ShapeNetRend(ShapeNetBase):
    """
    This class loads ShapeNetCore from a given directory into a Dataset object.
    ShapeNetCore is a subset of the ShapeNet dataset and can be downloaded from
    https://www.shapenet.org/.
    """

    def __init__(
        self,
        data_dir,
        split,
        object_class = "chair",
        dset_partition=-1,
        srf=None,
        use_lower_res=False

    ):
        """
        Store each object's synset id and models id from data_dir.
        Args:
            data_dir: Path to ShapeNetCore data.
            synsets: List of synset categories to load from ShapeNetCore in the form of
                synset offsets or labels. A combination of both is also accepted.
                When no category is specified, all categories in data_dir are loaded.
            version: (int) version of ShapeNetCore data in data_dir, 1 or 2.
                Default is set to be 1. Version 1 has 57 categories and verions 2 has 55
                categories.
                Note: version 1 has two categories 02858304(boat) and 02992529(cellphone)
                that are hyponyms of categories 04530566(watercraft) and 04401088(telephone)
                respectively. You can combine the categories manually if needed.
                Version 2 doesn't have 02858304(boat) or 02834778(bicycle) compared to
                version 1.
            load_textures: Boolean indicating whether textures should loaded for the model.
                Textures will be of type TexturesAtlas i.e. a texture map per face.
            texture_resolution: Int specifying the resolution of the texture map per face
                created using the textures in the obj file. A
                (texture_resolution, texture_resolution, 3) map is created per face.
        """
        super().__init__()
        self.split = split
        self.object_class = object_class
        self.dset_partition = dset_partition
        self.object_class_dir = os.path.join(data_dir, self.object_class)
        self.srf = srf
        self.nb_rf_variants = 1 # # TODO change to X when new SRF fukk rf_variants available  

        self.use_lower_res = use_lower_res
        # self.transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = torch.nn.Identity()


        #  transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # normalize,
        # ])
        splits = pd.read_csv(os.path.join(
            data_dir, "SNRL_splits.csv"), sep=",", dtype=str)
        avail_files = sorted(list(os.listdir(self.object_class_dir)))
        if self.dset_partition == -1:
            splits = splits[splits.file.isin(avail_files) & splits.classlabel.isin([str(self.object_class)])]

        else:
            splits = splits[splits.file.isin(avail_files) &  splits.partition.isin([str(x) for x in range(
                self.dset_partition+1)]) & splits.classlabel.isin([str(self.object_class)])]

        self.model_ids = list(splits[splits.split.isin([self.split])]["file"])
        # print(len(self.model_ids))
        self.synset_ids = [object_class for _ in range(len(self.model_ids))]

    def __getitem__(self, idx: int):
        """
        Read a model by the given index.
        Args:
            idx: The idx of the model to be retrieved in the dataset.
        Returns:
            dictionary with following keys:
            - verts: FloatTensor of shape (V, 3).
            - faces: LongTensor of shape (F, 3) which indexes into the verts tensor.
            - synset_id (str): synset id
            - model_id (str): model id
            - label (str): synset label.
        """

        model = self._get_item_ids(idx)
        rf_variant = np.random.randint(self.nb_rf_variants) 
        d_dir = os.path.join(self.object_class_dir, model["model_id"])
        model_path = os.path.join(d_dir, self.srf.relative_features_dir, "data_{}.npz".format(rf_variant))
        coords, feats = self.srf.load_coords_and_feats(model_path, device="cpu")
        feats = self.srf.normalize_rf(feats)
        labels = os.path.join(self.object_class_dir, model["model_id"])
        l_coords, l_feats = None,None
        if self.use_lower_res:
            l_relative_features_dir = os.path.join(self.srf.rf_alias, "vox{}".format(str(self.srf.available_vox_res[self.srf.available_vox_res.index(self.srf.vox_res) - 1])), self.srf.partial_alias)
            l_model_path = os.path.join(d_dir, l_relative_features_dir, "data_{}.npz".format(rf_variant))
            l_coords, l_feats = self.srf.load_coords_and_feats(l_model_path, device="cpu")
            l_coords, l_feats = self.srf.enlarge_sparse_voxels(l_coords, l_feats, factor=4)

        in_coords_, in_feat_, c2ws, imgs,masks, in_rf_variant = self.srf.reflection_function(d_dir, device="cpu", split=self.split)
        if self.srf.diffusion_type == "none":
            t = torch.Tensor([0])
        else:
            t = torch.randint(1, self.srf.time_steps, size=(1,)) if self.split == "train" else torch.Tensor([self.srf.time_steps-1])
        in_coords_, in_feat_, c2ws, cam_embed, imgs, t_embedd = self.srf.preprocess_input_rf(in_coords_, in_feat_, c2ws, imgs, t, transforms=self.transform)
        
        # DIFFUSION
        if self.srf.diffusion_type != "none":
            # if self.split == "train":
                # in_coords, in_feat = self.srf.forward_diffusion(coords, feats, t=t.item())
                # in_coords_, in_feat_ = self.srf.combine_two_srfs(in_coords_, in_feat_, in_coords, in_feat)
            in_coords, in_feat = torch.empty((self.srf.diffusion_kernel_size, 3)), torch.empty((self.srf.diffusion_kernel_size, self.srf.input_sh_dim*3+1))
            in_coords, in_feat = self.srf.diffusion_kernel(in_coords, in_feat, std=self.srf.kernel_std)
            if self.srf.ignore_input:
                in_coords_, in_feat_ = in_coords, in_feat
            else:
                in_coords_, in_feat_ = self.srf.combine_two_srfs(in_coords_, in_feat_, in_coords, in_feat)
            # coords, feats = self.srf.forward_diffusion(coords, feats, t=t.item()-1)

        return coords, feats, labels, in_coords_, in_feat_, c2ws,cam_embed, imgs, torch.from_numpy(masks)[None,...], l_coords, l_feats, t, t_embedd, torch.Tensor([in_rf_variant])
