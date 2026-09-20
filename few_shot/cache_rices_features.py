# Cache RICES

import os
import sys

import torch

from rices import RICES

sys.path.append('/home/user/Rizwan/Hate-Meme-Large-VLMs/hatevlms/data_set')
sys.path.append('/home/user/Rizwan/Hate-Meme-Large-VLMs/hatevlms')


class CacheRicesFeatures:
    def __init__(
            self,
            train_dataset,
            batch_size,
            embedding_details,
            output_pickle_file_name: str,
            cached_features_path=None
    ):
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.embedding_details = embedding_details
        self.output_pickle_file_name = output_pickle_file_name

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if cached_features_path is not None:
            cached_features = torch.load(
                os.path.join(self.embedding_details["output_dir_features"], self.output_pickle_file_name),
                map_location="cpu",
            )
        else:
            cached_features = None

        self.rices_dataset = RICES(
            dataset=self.train_dataset.get_dataset(),
            device=device,
            batch_size=self.batch_size,
            embedding_details=embedding_details,
            base_folders=self.train_dataset.get_base_folders(),
            cached_features=cached_features,
        )

    def cache_features(self):

        if not os.path.exists(self.embedding_details["output_dir_features"]):
            os.mkdir(self.embedding_details["output_dir_features"])

        torch.save(
            self.rices_dataset.features,
            os.path.join(self.embedding_details["output_dir_features"], self.output_pickle_file_name),
        )
