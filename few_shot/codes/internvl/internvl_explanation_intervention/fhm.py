#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install -U transformers accelerate


# In[2]:


import json
import os
import csv
import pandas as pd

from torch.utils.data import Dataset


class FacebookHatefulMemeDataset(Dataset):

    def __init__(self, base_folder, split):
        self.data_set = []

        dataset_path = os.path.join(base_folder, split + '.jsonl')
        with open(dataset_path, 'r') as json_file:
            json_list = list(json_file)

        for json_str in json_list:
            result = json.loads(json_str)
            self.data_set.append(result)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        return self.data_set[index]


class MAMIDataset(Dataset):
    def __init__(self, base_folder, split):
        self.data_set = []

        dataset_path = os.path.join(base_folder, split + '.tsv')

        # Open and parse the TSV file, directly populating self.data_set
        with open(dataset_path, 'r') as tsv_file:
            reader = csv.DictReader(tsv_file, delimiter='\t')

            for row in reader:
                data_entry = {
                    "img": "images/" + row['file_name'],
                    "label": int(row['label']),
                    "text": row['text']
                }
                self.data_set.append(data_entry)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        return self.data_set[index]


class Harm_C_Dataset(Dataset):
    def __init__(self, base_folder, split):
        self.data_set = []

        dataset_path = os.path.join(base_folder, split + '.jsonl')
        with open(dataset_path, 'r') as json_file:
            json_list = list(json_file)

        for json_str in json_list:
            result = json.loads(json_str)
            result['label'] = 0 if 'not harmful' in result['labels'] else 1
            result['img'] = 'images/' + result['image']
            self.data_set.append(result)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        return self.data_set[index]


class Harm_P_Dataset(Dataset):
    def __init__(self, base_folder, split):
        self.data_set = []

        dataset_path = os.path.join(base_folder, split + '.jsonl')
        with open(dataset_path, 'r') as json_file:
            json_list = list(json_file)

        for json_str in json_list:
            result = json.loads(json_str)
            result['label'] = 0 if 'not harmful' in result['labels'] else 1
            result['img'] = 'images/' + result['image']
            self.data_set.append(result)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        return self.data_set[index]


class BHMDataset(Dataset):
    def __init__(self, base_folder, split):
        self.data_set = []

        dataset_path = os.path.join(base_folder, 'Files', split + '_task1_translated_indictrans2.xlsx')

        data = pd.read_excel(dataset_path)

        for _, row in data.iterrows():
            label = 1 if row['Labels'] == 'hate' else 0

            data_entry = {
                "img": os.path.join(base_folder, "Memes", row['image_name']),
                "text": row['Captions_English'],
                "label": label  # Hate (1) or Non-hate (0)
            }
            self.data_set.append(data_entry)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        return self.data_set[index]


# In[3]:


from torch.utils.data import ConcatDataset


class DatasetWrapper:
    def __init__(self, dataset_tag, base_folders, split):
        self.base_folders = base_folders
        if dataset_tag == "facebook_hateful_meme_dataset":
            self.dataset = FacebookHatefulMemeDataset(
                base_folder=base_folders[0],
                split=split
            )
        elif dataset_tag == "MAMI_dataset":
            self.dataset = MAMIDataset(
                base_folder=base_folders[0],
                split=split
            )
        elif dataset_tag == "Harm_P_Dataset":
            self.dataset = Harm_P_Dataset(
                base_folder=base_folders[0],
                split=split
            )
        elif dataset_tag == "Harm_C_Dataset":
            self.dataset = Harm_C_Dataset(
                base_folder=base_folders[0],
                split=split
            )
        elif dataset_tag == "combined_Harm_CP_Dataset":
            if len(base_folders) != 2:
                raise ValueError("Both base_folder_c and base_folder_p must be provided for combined dataset.")
            self.dataset = DatasetWrapper.__get_combined_dataset(
                base_folders[0],
                base_folders[1],
                split
            )
        elif dataset_tag == "BHM_Dataset":
            self.dataset = BHMDataset(
                base_folder=base_folders[0],
                split=split
            )

    def get_dataset(self):
        return self.dataset

    def get_base_folders(self):
        return self.base_folders

    @staticmethod
    def __get_combined_dataset(base_folder_c, base_folder_p, split):
        covid_dataset = Harm_C_Dataset(
            base_folder_c,
            split
        )
        politics_dataset = Harm_P_Dataset(
            base_folder_p,
            split
        )
        return ConcatDataset([covid_dataset, politics_dataset])


# In[4]:


# Supported Embeddings
EMBEDDINGS = [
    "clip_image",
    "blip_image",
    "sig_lip_image",
    "image_bind_image",
    "clip_image_text",
    "blip_image_text",
    "sig_lip_image_text",
    "image_bind_image_text",
    "custom"
]

# Meta Data Map
EMBEDDING_MAP = {
    # CLIP Image Encoder
    EMBEDDINGS[0]: {
        # https://codeandlife.com/2023/01/26/mastering-the-huggingface-clip-model-how-to-extract-embeddings-and-calculate-similarity-for-text-and-images/
        # https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPModel.get_image_features
        "index": 0,
        "image_encoder": "openai/clip-vit-large-patch14",
        "output_dir_features": "/home/subhankar-am/nrizwan/experiments/embeddings/embeddings/clip_image"
    },

    # BLIP Image Encoder
    EMBEDDINGS[1]: {
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py#L582
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py#L1335
        # https://huggingface.co/docs/transformers/en/model_doc/blip-2#transformers.Blip2Model.get_image_features
        "index": 1,
        "image_encoder": "Salesforce/blip2-flan-t5-xl",
        "output_dir_features": "/home/subhankar-am/nrizwan/experiments/embeddings/embeddings/blip_image"
    },

    # SIG_LIP Image Encoder
    EMBEDDINGS[2]: {
        # https://huggingface.co/docs/transformers/en/model_doc/siglip#transformers.SiglipModel.get_image_features
        "index": 2,
        "image_encoder": "google/siglip-large-patch16-256",
        "output_dir_features": "/home/subhankar-am/nrizwan/experiments/embeddings/embeddings/sig_lip_image"
    },

    # Image_Bind Image Encoder
    EMBEDDINGS[3]: {
        "index": 3,
        "image_encoder": "",
        "output_dir_features": "/home/subhankar-am/nrizwan/experiments/embeddings/embeddings/image_bind_image"
    },

    # CLIP Image Text Encoder
    EMBEDDINGS[4]: {
        # https://codeandlife.com/2023/01/26/mastering-the-huggingface-clip-model-how-to-extract-embeddings-and-calculate-similarity-for-text-and-images/
        # https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPModel.get_image_features
        # https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPModel.get_text_features
        "index": 4,
        "image_encoder": "openai/clip-vit-large-patch14",
        "text_encoder": "openai/clip-vit-large-patch14",
        "output_dir_features": "/home/subhankar-am/nrizwan/experiments/embeddings/embeddings/clip_image_text"
    },

    # BLIP Image Text Encoder
    EMBEDDINGS[5]: {
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py#L582
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py#L1335
        # https://huggingface.co/docs/transformers/en/model_doc/blip-2#transformers.Blip2Model.get_text_features
        # https://huggingface.co/docs/transformers/en/model_doc/blip-2#transformers.Blip2Model.get_image_features
        "index": 5,
        "image_encoder": "Salesforce/blip2-flan-t5-xl",
        "text_encoder": "Salesforce/blip2-flan-t5-xl",
        "output_dir_features": "/home/subhankar-am/nrizwan/experiments/embeddings/embeddings/blip_image_text"
    },

    # SIG_LIP Image Text Encoder
    EMBEDDINGS[6]: {
        # https://huggingface.co/docs/transformers/en/model_doc/siglip#transformers.SiglipModel.get_text_features
        # https://huggingface.co/docs/transformers/en/model_doc/siglip#transformers.SiglipModel.get_image_features
        "index": 6,
        "image_encoder": "google/siglip-large-patch16-256",
        "text_encoder": "google/siglip-large-patch16-256",
        "output_dir_features": "/home/subhankar-am/nrizwan/experiments/embeddings/embeddings/sig_lip_image_text"
    },

    # Image_Bind Image Text Encoder
    EMBEDDINGS[7]: {
        "index": 7,
        "image_encoder": "",
        "text_encoder": "",
        "output_dir_features": "/home/subhankar-am/nrizwan/experiments/embeddings/embeddings/image_bind_image_text"
    },

    # Custom Embedding Generator
    EMBEDDINGS[8]: {
        "index": 8,
        "image_encoder": "",
        "text_encoder": "",
        "output_dir_features": "/home/subhankar-am/nrizwan/experiments/embeddings/embeddings/custom"
    },
}


# In[5]:


def custom_collate_fn(batch):
    """
    Collate function for DataLoader that collates a list of dicts into a dict of lists.
    """
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch


# In[6]:


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
        print(self.device)
        self.model.to("cuda:0")

        self.processor = AutoProcessor.from_pretrained(self.embedding_details["image_encoder"])

    def __model_specific_image_feature_computation(self, image_features):
        if self.embedding_details["index"] in [1, 5]:
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py#L582
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py#L1335
            return image_features.pooler_output

        return image_features

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
                    pass
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
            ).to("cuda:0")

            query_feature_image = self.model.get_image_features(**inputs)
            query_feature_image = self.__model_specific_image_feature_computation(query_feature_image)

            if query_feature_image.ndim == 1:
                query_feature_image = query_feature_image.unsqueeze(0)

            query_feature_image /= query_feature_image.norm(dim=-1, keepdim=True)

            # Compute query features from image/text embeddings
            if self.embedding_details["index"] in [0, 1, 2, 3]:
                query_feature = query_feature_image
            elif self.embedding_details["index"] in [4, 5, 6, 7]:
                pass
            else:
                assert self.embedding_details["index"] in [8]

            query_feature = query_feature.detach().cpu()
            print("Final Features:", query_feature.shape, len(query_feature), self.features.shape)

            # Compute the similarity of the input image to the precomputed features
            similarity = (query_feature @ self.features.T).squeeze()

            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)

            # Get the indices of the 'num_examples' most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :num_examples]

        # Return with the most similar images last
        return [[self.dataset[i] for i in reversed(row)] for row in indices]


# In[7]:


# def load_jsonl_folder(folder_path, key="img"):
#     """
#     Load all JSONL files in a folder into a dictionary indexed by the specified key.
#     """
#     data_map = {}
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith(".jsonl"):
#             file_path = os.path.join(folder_path, file_name)
#             with open(file_path, "r") as f:
#                 for line in f:
#                     entry = json.loads(line)
#                     data_map[entry[key]] = entry
#     return data_map

def load_json_folder(folder_path, key="img_fname", value="generated_caption"):
    """
    Load all JSON files in a folder into a dictionary indexed by the specified key.
    Each file is assumed to contain a JSON array of objects.
    """
    data_map = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                entries = json.load(f)  # load the entire JSON array

                if not isinstance(entries, list):
                    raise ValueError(f"{file_name} does not contain a JSON array")

                for entry in entries:
                    k = entry.get(key)
                    if k is not None:
                        # if k in data_map:
                        #     raise ValueError(f"Duplicate key '{k}' found in {file_path}")
                        data_map[k] = entry.get(value)
    return data_map

# def get_precomputed_data(img_path, generated_captions_map, generated_explanations_map):
#     """
#     Get precomputed caption and explanation for an image.
#     """
#     generated_captions_entry = generated_captions_map.get(img_path, {})
#     generated_explanations_entry = generated_explanations_map.get(img_path, {})
    
#     return {
#         "caption": generated_captions_entry.get("generated_caption", ""),
#         "explanation": generated_explanations_entry.get("generated_explanation", ""),
#         "label": generated_explanations_entry.get("label", -1)
#     }


# In[8]:


# Paths to folders containing captions and explanations
generated_captions_folder = "/home/subhankar-am/nrizwan/experiments/generated-captions"
generated_explanations_folder = "/home/subhankar-am/nrizwan/experiments/generated-explanations"
generated_intervention_folder = "/home/subhankar-am/nrizwan/experiments/intervention"

# Load precomputed captions and explanations
generated_captions_map = load_json_folder(generated_captions_folder, key="img_fname")
# generated_explanations_map = load_json_folder(generated_explanations_folder, key="img_fname")
generated_intervention_map = load_json_folder(generated_intervention_folder, key="img", value="intervention")


# In[9]:


import csv

# Path to your CSV file  
fhm_exp_file_path = '/home/subhankar-am/nrizwan/experiments/fhm-explanation/Annotation of Explanation (test) - FHM-train-not-hateful.csv'

generated_interpretations_map = {}

with open(fhm_exp_file_path, 'r', newline='', encoding='utf-8') as csvfile:  
    csvreader = csv.reader(csvfile)  
    header = next(csvreader)  # skip header row
    for row in csvreader:
        img_path = row[1]  # "img/xxxxx.png"
        explanation = row[3]  # "explanation" column (4th column, index=3)
        
        assert img_path not in generated_interpretations_map
        generated_interpretations_map[img_path] = explanation.strip()

# print(list(generated_interpretations_map.items())[:5])  # print first 5 entries


# In[10]:


# from PIL import Image
# import json
# import os
# from tqdm import tqdm
# import torch
# import copy
# import base64
# from IPython.display import display
# import io


# class IDEFICSInference:

#     def __init__(self, checkpoint):
#         self.checkpoint = checkpoint
#         self.max_new_tokens = 50

#     def encode_image(self, image_path):
#         with open(image_path, "rb") as image_file:
#             return base64.b64encode(image_file.read()).decode('utf-8')
    
#     def generate_output(
#             self,
#             base_folders,
#             image_metadata,
#             output_file,
#             rices_class_few_shot,
#             shots,
#             dictionary,
#             temperature
#     ):
#         selected_images = []

#         if rices_class_few_shot is not None:
#             selected_images = rices_class_few_shot.find([image_metadata], shots)

#         # Collect images and build aligned text prompt
#         all_images = [None]
#         text_prompt_parts = []

#         # System/instruction
#         system_msg = (
#             "User: Consider the following definitions.\n"
#             "1. hateful - a direct or indirect attack on people based on characteristics, including ethnicity, race, "
#             "nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, "
#             "and disability or disease. Attack is defined as violent or dehumanizing (comparing people to non-human "
#             "things, e.g. animals) speech, statements of inferiority, and calls for exclusion or segregation. Mocking "
#             "hate crime is also considered hateful.\n"
#             "2. not-hateful - an attack which is not hateful and follows social norms.\n"
#             "Classify the meme as hateful or not-hateful based on the above definitions considering the image "
#             "as well as the extracted text from the image delimited by three backticks.\n"
#             "Example output for hateful meme : hateful\n"
#             "Example output for not hateful meme : not-hateful<end_of_utterance>"
#         )
#         text_prompt_parts.append(system_msg)

#         # Few-shot examples
#         for img_data in selected_images[0] if selected_images else []:
#             extracted_text = img_data["text"]
#             image_path = self._get_image(img_data["img"], base_folders)
#             label_out = dictionary[img_data["label"]]

#             all_images.append(Image.open(image_path))

#             text_prompt_parts.append(
#                 f"\nUser: <image>\nExtracted text from the image: ```{extracted_text}```<end_of_utterance>"
#                 f"\nAssistant: {label_out}<end_of_utterance>"
#             )

#         # Current query
#         extracted_text = image_metadata["text"]
#         image_path = self._get_image(image_metadata["img"], base_folders)

#         all_images.append(Image.open(image_path))

#         text_prompt_parts.append(
#             f"\nUser: <image>\nExtracted text from the image: ```{extracted_text}```<end_of_utterance>\nAssistant:"
#         )

#         # Process inputs cleanly
#         inputs = self.checkpoint.processor(
#             text=text_prompt_parts,
#             images=all_images,
#             return_tensors="pt"
#         )
#         inputs = {k: v.to(self.checkpoint.model.device) for k, v in inputs.items()}

#         # Generation args
#         exit_condition = self.checkpoint.processor.tokenizer(
#             "<end_of_utterance>", add_special_tokens=False
#         ).input_ids
#         bad_words_ids = self.checkpoint.processor.tokenizer(
#             ["<image>", "<fake_token_around_image>"], add_special_tokens=False
#         ).input_ids

#         # Generate
#         with torch.no_grad():
#             generated_ids = self.checkpoint.model.generate(
#                 **inputs,
#                 eos_token_id=exit_condition,
#                 bad_words_ids=bad_words_ids,
#                 max_new_tokens=self.max_new_tokens,
#                 temperature=temperature,
#                 do_sample=temperature > 0
#             )

#         generated_texts = self.checkpoint.processor.batch_decode(
#             generated_ids, skip_special_tokens=True
#         )
#         generated_text = generated_texts[0] if generated_texts else ""

#         output_data = {
#             "id": image_metadata.get("id", "unknown"),
#             "img": image_metadata["img"],
#             "label": image_metadata.get("label"),
#             "text": image_metadata["text"],
#             "output": generated_text
#         }

#         print(output_data)
#         print("#########################################################")

#         output_file.write(json.dumps(output_data) + "\n") 


#     def _get_image(self, image_path, base_folders):
#         for base_folder in base_folders:
#             full_path = os.path.join(base_folder, image_path)
#             if os.path.exists(full_path):
#                 return full_path
        
#         # If image is not found in any of the base folders, raise an error or return None as per your requirement
#         raise FileNotFoundError(f"Image '{image_path}' not found in any of the base folders.")


# In[11]:




# In[12]:


from PIL import Image
import os
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


class InternVLInference:

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

        self.max_new_tokens = 100

    def generate_output(
        self,
        base_folders,
        image_metadata,
        output_file,
        rices_class_few_shot,
        shots,
        dictionary,
        temperature,
    ):
        selected_images = []

        if rices_class_few_shot is not None:
            selected_images = rices_class_few_shot.find([image_metadata], shots)

        messages = self.generate_few_shot_prompts(selected_images, dictionary, base_folders)

        caption_img = generated_captions_map.get(image_metadata["img"], "")

        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": self._get_image(image_metadata["img"], base_folders)},
                {"type": "text", "text": f"\nCaption of the image: {caption_img}"},
                {"type": "text", "text": f"""\nExtracted text from the image: ```{image_metadata['text']}```"""},
            ],
        })
        
        # print(messages)

        # Preprocess input
        text = self.checkpoint.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.checkpoint.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda:0")

        # Generate
        generated_ids = self.checkpoint.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        generated_texts = self.checkpoint.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for text in generated_texts:
            print(text)
            output_file.writelines(text + "\n")

    def generate_few_shot_prompts(self, selected_images, dictionary, base_folders):
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text":
                        "You are an AI assistant tasked with classifying memes into hateful or not-hateful along with an explanation while also generating an intervention only for hateful memes based on the input image, image caption and extracted text obtained from the image.\n"
                        "Consider the following definitions.\n"
                        "1. hateful - a direct or indirect attack on people based on characteristics, including ethnicity, race, "
                        "nationality, immigration status, religion, caste, sex, gender identity, sexual orientation, "
                        "and disability or disease. Attack is defined as violent or dehumanizing (comparing people to non-human "
                        "things, e.g. animals) speech, statements of inferiority, and calls for exclusion or segregation. Mocking "
                        "hate crime is also considered hateful.\n"
                        "2. not-hateful - an attack which is not hateful and follows social norms.\n"
                        "Classify the meme as hateful or not-hateful and then explain the reason based on the above definitions considering the image, provided caption of the image, "
                        "as well as the extracted text from the image delimited by three backticks. If the meme is hateful then also generate an intervention for it.\n"
                        "Example output for hateful meme : hateful - explanation in 30 words that why the meme is hateful. Intervention - intervention in 30 words for the hateful meme.\n"
                        "Example output for not hateful meme : not-hateful - explanation in 30 words that why the meme is not-hateful."
                    }
                ],
            }
        ]

        if selected_images:
            for i in range(len(selected_images[0])):
                caption_img = generated_captions_map.get(selected_images[0][i]["img"], "")
                interpretation_img = generated_interpretations_map.get(selected_images[0][i]["img"], "")  

                # Few-shot examples
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": self._get_image(selected_images[0][i]["img"], base_folders)},
                        {"type": "text", "text": f"\nCaption of the image: {caption_img}"},
                        {"type": "text", "text": f"""\nExtracted text from the image: ```{selected_images[0][i]['text']}```"""},
                    ],
                })
                
                intervene = ""
                if selected_images[0][i]["label"] == 1 and selected_images[0][i]["img"] in generated_intervention_map:
                    intervene = "\nIntervention - " + generated_intervention_map[selected_images[0][i]["img"]]
                
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"{dictionary[selected_images[0][i]['label']]} - {interpretation_img}{intervene}"}],
                })

        return messages

    def _get_image(self, image_path, base_folders):
        for base_folder in base_folders:
            full_path = os.path.join(base_folder, image_path)
            if os.path.exists(full_path):

                return Image.open(full_path)
        raise FileNotFoundError(f"Image '{image_path}' not found in any base folder.")




import os
import os.path
import json

from tqdm import tqdm
import torch
from transformers import (
    AutoProcessor,
    AutoModel,
    Blip2Model,
    CLIPModel
)

class PerformInference:
    def __init__(
            self,
            dataset_tag,
            base_folders,
            split,
            output_pickle_file_name,
            use_rices_feature,
            embed_details,
            model_checkpoint
    ):
        self.dataset_tag = dataset_tag
        self.base_folders = base_folders

        # Initialize dataset wrapper class
        self.dataset_wrapper = DatasetWrapper(
            dataset_tag=self.dataset_tag,
            base_folders=self.base_folders,
            split=split
        )

        # Initialize checkpoint initializer class
        self.checkpoint_initializer = InternVLCheckpointInitializer(model_checkpoint)

        self.rices_few_shot = None
        if use_rices_feature:
            self.__few_shot_setup(embed_details["output_dir_features"], output_pickle_file_name, embed_details)

    def __few_shot_setup(self, output_dir_features: str, output_pickle_file_name: str, embed_details):
        # Initialize RICES class
        train_dataset_wrapper = DatasetWrapper(
            dataset_tag=self.dataset_tag,
            base_folders=self.base_folders,
            split="train"
        )

        cached_features = torch.load(
            os.path.join(output_dir_features, output_pickle_file_name),
            map_location="cpu"
        )

        self.rices_few_shot = RICES(
            dataset=train_dataset_wrapper.get_dataset(),
            device="cuda:1" if torch.cuda.is_available() else "cpu",
            batch_size=256,
            embedding_details=embed_details,
            base_folders=train_dataset_wrapper.get_base_folders(),
            cached_features=cached_features
        )

    def generate_output(
            self,
            output_directory,
            output_filename,
            labels_dictionary,
            shots,
            number_of_iterations=1,
            temperature = 0.001
    ):
        for iteration in range(number_of_iterations):
            output_file_path = os.path.join(
                output_directory,
                output_filename + "_" + str(iteration) + '.txt'
            )

            output_file = open(output_file_path, 'w')
            counter = 0
            for image in tqdm(self.dataset_wrapper.dataset):
                # counter += 1
                # if counter >= 2:
                #     continue
                output_file.writelines(json.dumps(image))
                output_file.write('\n----------\n')
                self.checkpoint_initializer.inference.generate_output(
                    self.base_folders,
                    image,
                    output_file,
                    self.rices_few_shot,
                    shots,
                    labels_dictionary,
                    temperature
                )
                output_file.write("\n##########\n")
            output_file.close()

    def dynamically_update_inference_class(self):
        self.checkpoint_initializer.update_inference_class()


# In[14]:


from transformers import AutoProcessor, AutoModelForImageTextToText


class InternVLCheckpointInitializer:

    def __init__(self, model_checkpoint="OpenGVLab/InternVL3_5-8B-HF"):
        # Load model
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_checkpoint,
            torch_dtype=torch.bfloat16,   
            device_map="auto"     
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_checkpoint)
        
        # Attach inference class
        self.inference = InternVLInference(self)

    def update_inference_class(self):
        self.inference = InternVLInference(self)


# In[15]:


import email, smtplib, ssl, os

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def email_classification_report(folder, filename,receiver_email):
    sender_email = "testemailskgp@gmail.com"
    subject = "Classification Report of "+ filename

    filepath = os.path.join(folder, filename)

    body = ""

    # password = input("Type your password and press enter:")
    password = "yqnm mjus lxil mcbp"

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message["Bcc"] = receiver_email

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    with open(filepath, 'rb') as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email    
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {filename}",
    )

    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)


# In[16]:

import gc
for embedding in [EMBEDDINGS[2]]:

    embedding_details = EMBEDDING_MAP[
        embedding
    ]
    
    perform_inference = PerformInference(
        dataset_tag="facebook_hateful_meme_dataset",
        base_folders=["/home/subhankar-am/nrizwan/experiments/facebook-hateful-memes/archive (1)/hateful_memes/hateful_memes"],
        split="test_seen",
        output_pickle_file_name="FHM.pkl",
        use_rices_feature=True,
        embed_details=embedding_details,
        model_checkpoint="OpenGVLab/InternVL3_5-8B-HF"
    )
    perform_inference.dynamically_update_inference_class()
    
    # perform_inference = PerformInference(
    #     dataset_tag="MAMI_dataset",
    #     base_folders=["/kaggle/input/mami-memes/mami-dataset"],
    #     split="test",
    #     output_pickle_file_name="MAMI.pkl",
    #     use_rices_feature=True,
    #     embed_details=embedding_details,
    #     model_checkpoint="gpt-4o-hate-alert"
    # )
    
    # perform_inference = PerformInference(
    #     dataset_tag="combined_Harm_CP_Dataset",
    #     base_folders=["/kaggle/input/harm-c-memes-dataset", "/kaggle/input/harm-p-memes-dataset"],
    #     split="test",
    #     output_pickle_file_name="HARM_CP.pkl",
    #     use_rices_feature=True,
    #     embed_details=embedding_details,
    #     model_checkpoint="gpt-4o-hate-alert"
    # )

    for shot in [2, 4, 8]:
        file_name = f"internvl_RICES_{embedding}_{shot}_shots_FHM_test_seen"
        
        perform_inference.generate_output(
            output_directory="/home/subhankar-am/nrizwan/experiments/internvl_explanation_intervention/",
            output_filename=file_name,
            labels_dictionary={
                0: "not-hateful",
                1: "hateful"
            },
            shots=shot
        )
    del perform_inference
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        ##email_classification_report('/home/subhankar-am/nrizwan/experiments/qwen_explanation/', f"{file_name}_0.txt", "subhankar.official185@gmail.com")


# In[ ]:





# In[ ]:




