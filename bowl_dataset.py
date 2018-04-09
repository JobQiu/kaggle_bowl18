from utils import Dataset
from glob import glob
import os
import numpy as np
import re
import cv2



class BowlDataset(Dataset):


    def load_bowl(self, base_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        problem_ids = list()
        problem_ids.append('7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80')
        problem_ids.append('b1eb0123fe2d8c825694b193efb7b923d95effac9558ee4eaf3116374c2c94fe')
        problem_ids.append('9bb6e39d5f4415bc7554842ee5d1280403a602f2ba56122b87f453a62d37c06e')
        problem_ids.append('1f0008060150b5b93084ae2e4dabd160ab80a95ce8071a321b80ec4e33b58aca')
        problem_ids.append('58c593bcb98386e7fd42a1d34e291db93477624b164e83ab2afa3caa90d1d921')
        problem_ids.append('adc315bd40d699fd4e4effbcce81cd7162851007f485d754ad3b0472f73a86df')
        problem_ids.append('12aeefb1b522b283819b12e4cfaf6b13c1264c0aadac3412b4edd2ace304cb40')
        problem_ids.append('0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9')

        self.add_class("bowl", 1, "nuclei")

        masks = dict()
        id_extractor = re.compile(f"{base_path}\{os.sep}(?P<image_id>.*)\{os.sep}masks\{os.sep}(?P<mask_id>.*)\.png")

        for mask_path in glob(os.path.join(base_path, "**", "masks", "*.png")):
            matches = id_extractor.match(mask_path)

            image_id = matches.group("image_id")
            image_path = os.path.join(base_path, image_id, "images", image_id + ".png")

            if image_path in masks:
                masks[image_path].append(mask_path)
            else:
                masks[image_path] = [mask_path]

        for i, (image_path, mask_paths) in enumerate(masks.items()):
            if not image_path.split('/')[1] in problem_ids:
                self.add_image("bowl", image_id=i, path=image_path, mask_paths=mask_paths)


    def load_image(self, image_id):
        info = self.image_info[image_id]

        return cv2.imread(info["path"])


    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)


    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask_paths = info["mask_paths"]
        count = len(mask_paths)
        masks = []

        for i, mask_path in enumerate(mask_paths):
            masks.append(cv2.imread(mask_path, 0))

        masks = np.stack(masks, axis=-1)
        masks = np.where(masks > 128, 1, 0)
        
        class_ids = np.ones(count)
        return masks, class_ids.astype(np.int32)
