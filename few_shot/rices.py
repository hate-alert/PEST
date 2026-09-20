import os

import torch
from PIL import Image
from torch.utils import data
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoModel,
    Blip2Model,
    CLIPModel,
    AutoTokenizer
)


from .utils import custom_collate_fn


class RICES:
    def __init__(
            self,
            dataset,
            device,
            batch_size,
            embedding_details,
            base_folders,
            cached_features=None,

    ):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.embedding_details = embedding_details
        self.base_folders = base_folders

        # Set up the model and processor
        self.__setup_model_and_processors()

        # Precompute features
        if cached_features is None:
            self.features = self.__precompute_features()
        else:
            self.features = cached_features

    def __find_image_path(self, image):
        for base_folder in self.base_folders:
            full_path = os.path.join(base_folder, image)
            if os.path.exists(full_path):
                return full_path
        raise FileNotFoundError(f"Image '{image}' not found in any of the base folders.")

    def __setup_model_and_processors(self):
        print(self.dataset, end="\n----------\n")
        print(self.embedding_details, end="\n----------\n")

        if self.embedding_details["index"] in [0, 4]:
            self.model = CLIPModel.from_pretrained(self.embedding_details["image_encoder"])
        elif self.embedding_details["index"] in [1, 5]:
            self.model = Blip2Model.from_pretrained(
                self.embedding_details["image_encoder"],
                torch_dtype=torch.bfloat16
            )
        elif self.embedding_details["index"] in [2, 6]:
            self.model = AutoModel.from_pretrained(self.embedding_details["image_encoder"])

        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.embedding_details["image_encoder"])
        if self.embedding_details["index"] in [4, 5, 6, 7]:
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_details["text_encoder"])

    def __model_specific_image_feature_computation(self, image_features):
        if self.embedding_details["index"] in [1, 5]:
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py#L582
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py#L1335
            return image_features.pooler_output

        return image_features

    def __model_specific_text_feature_computation(self, text_features):
        if self.embedding_details["index"] in [1, 5]:
            # TODO
            pass

        return text_features

    def __precompute_features(self):
        features = []

        # Switch to evaluation mode
        self.model.eval()

        # Set up loader
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=custom_collate_fn,
        )

        with torch.no_grad():
            for batch in tqdm(
                    loader,
                    desc="Precomputing features for RICES",
            ):
                # Get the feature of the input image
                image_batch = batch["img"]

                inputs = self.processor(
                    images=[Image.open(self.__find_image_path(image)).convert('RGB') for image in image_batch],
                    return_tensors="pt"
                ).to(self.device)

                image_features = self.model.get_image_features(**inputs)
                image_features = self.__model_specific_image_feature_computation(image_features)

                image_features /= image_features.norm(dim=-1, keepdim=True)

                print("Image Features:", image_features.shape)

                # Compute query features from image/text embeddings
                if self.embedding_details["index"] in [0, 1, 2, 3]:
                    final_features = image_features
                elif self.embedding_details["index"] in [4, 5, 6, 7]:
                    # Get the feature of the input text
                    text_batch = batch["text"]

                    inputs = self.tokenizer(
                        [text for text in text_batch],
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device)


                    textual_features = self.model.get_text_features(**inputs)
                    textual_features = self.__model_specific_text_feature_computation(
                        textual_features
                    )

                    textual_features /= textual_features.norm(dim=-1, keepdim=True)

                    print("Textual Features:", textual_features.shape)
                    final_features = torch.cat((image_features, textual_features), -1)
                else:
                    assert self.embedding_details["index"] in [8]

                features.append(final_features.detach())
                print("Final Features:", final_features.shape, len(features))

        features = torch.cat(features)

        print(len(features), len(features[0]))
        return features

    def find(self, batch, num_examples):
        """
        Get the top num_examples most similar examples to the images.
        """
        # Switch to evaluation mode
        self.model.eval()

        with torch.no_grad():
            # Get the feature of the input image
            inputs = self.processor(
                images=[Image.open(self.__find_image_path(image['img'])).convert('RGB') for image in batch],
                return_tensors="pt"
            ).to(self.device)

            query_feature_image = self.model.get_image_features(**inputs)
            query_feature_image = self.__model_specific_image_feature_computation(query_feature_image)

            if query_feature_image.ndim == 1:
                query_feature_image = query_feature_image.unsqueeze(0)

            query_feature_image /= query_feature_image.norm(dim=-1, keepdim=True)

            print("Image Features:", query_feature_image.shape)

            # Compute query features from image/text embeddings
            if self.embedding_details["index"] in [0, 1, 2, 3]:
                query_feature = query_feature_image
            elif self.embedding_details["index"] in [4, 5, 6, 7]:
                # Get the feature of the input text
                inputs = self.tokenizer(
                    [image["text"] for image in batch],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                query_feature_text = self.model.get_text_features(**inputs)
                query_feature_text = self.__model_specific_text_feature_computation(query_feature_text)

                if query_feature_text.ndim == 1:
                    query_feature_text = query_feature_text.unsqueeze(0)

                query_feature_text /= query_feature_text.norm(dim=-1, keepdim=True)

                print("Textual Features:", query_feature_text.shape)
                query_feature = torch.cat((query_feature_image, query_feature_text), -1)
            else:
                assert self.embedding_details["index"] in [8]

            query_feature = query_feature.detach().cpu()
            print(
                "Final Features:",
                query_feature.shape,
                len(query_feature),
                self.features.shape,
            )

            # Compute the similarity of the input image to the precomputed features
            similarity = (query_feature @ self.features.T).squeeze()

            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)

            # Get the indices of the 'num_examples' most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :num_examples]

        # Return with the most similar images last
        return [[self.dataset[i] for i in reversed(row)] for row in indices]
