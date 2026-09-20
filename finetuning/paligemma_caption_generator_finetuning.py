from huggingface_hub import login

# Login to Hugging Face Hub
login()

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
from io import BytesIO
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
import numpy as np

import evaluate
from evaluate import load
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from tqdm import tqdm

# Define global DTYPE for consistent usage
DTYPE = torch.bfloat16

class MemeDataset(Dataset):
    def __init__(self, json_file, img_dir, split="train"):
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        # Filter data based on split
        # Assuming first 80% for training, rest for validation
        total = len(self.data)
        split_idx = int(0.8 * total)

        if split == "train":
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]

        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        img_path = os.path.join(self.img_dir, item["img_fname"])
        # If image not found, download it from URL (if available)
        if not os.path.exists(img_path):
            image_url = item.get("url", None)
            if image_url:
                try:
                    response = requests.get(image_url, stream=True)
                    if response.status_code == 200:
                        os.makedirs(os.path.dirname(img_path), exist_ok=True)
                        with open(img_path, "wb") as f:
                            for chunk in response.iter_content(1024):
                                f.write(chunk)
                    else:
                        print(f"Failed to download: {image_url}")
                except Exception as e:
                    print(f"Error downloading image: {e}")

        if not os.path.exists(img_path):
            return None
                 
        image = Image.open(img_path).convert("RGB")

        # Get caption (using the first meme caption)
        prompt = "Caption the given image."
        caption = item["meme_captions"][0] if item["meme_captions"] else ""


        return (image, prompt, caption)

# from transformers import BitsAndBytesConfig
# from peft import get_peft_model, LoraConfig

# bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

# lora_config = LoraConfig(
#     r=8,
#     target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
#     task_type="CAUSAL_LM",
# )


def train_meme_caption_model(json_file, img_dir, output_dir):
    # Load model and processor
    model_name = "google/paligemma-3b-pt-448"
    processor = AutoProcessor.from_pretrained(model_name)

    ## If you want Full Fine Tuning
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_name, torch_dtype=DTYPE, device_map="auto")

    for param in model.vision_tower.parameters():
        param.requires_grad = False

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False

    ## If you want QLoRa Fine Tuning
    # model = PaliGemmaForConditionalGeneration.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config)
    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()


    # Create datasets
    train_dataset = MemeDataset(json_file, img_dir, split="train")
    eval_dataset = MemeDataset(json_file, img_dir, split="val")

    for image, text, label in train_dataset:
        print(image, text, label)
        break

    image_token = processor.tokenizer.convert_tokens_to_ids("<image>")
    def collate_fn(examples):
        examples = [example for example in examples if example is not None]

        if len(examples) == 0:
            return None  # Handle empty batch
        
        images = [ex[0] for ex in examples]
        prompts = ["<image>" + ex[1] for ex in examples]
        labels = [ex[2] for ex in examples]
        tokens = processor(
            text=prompts,
            images=images,
            suffix=labels,
            return_tensors="pt",
            padding="longest"
        )
        # print(model.device)
        # tokens = tokens.to(DTYPE).to(model.device)
        return tokens

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        eval_steps=1000,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        warmup_steps=2,
        learning_rate=2e-5,
        weight_decay=1e-6,
        adam_beta2=0.999,
        # optim="adamw_hf",
        fp16=False,  # Set to False since we're using bfloat16
        bf16=True,   # Use bfloat16 instead
        num_train_epochs=1,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        dataloader_pin_memory=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    # Train model
    trainer.train()
    # trainer.train(resume_from_checkpoint=True)

    return model, processor

if __name__ == "__main__":
    # Train model
    model, processor = train_meme_caption_model(
        "./meme_caption_dataset/memes-trainval.json",
        "./meme_caption_dataset/memes",
        "meme-caption-model",
    )

    # Save model and processor
    model.save_pretrained(f"meme-caption-model/final")
    processor.save_pretrained(f"meme-caption-model/final")

model.push_to_hub("pbhaskar/meme-caption-generator")
processor.push_to_hub("pbhaskar/meme-caption-generator")

# Clear GPU Memory
import gc

gc.collect()
torch.cuda.empty_cache()

for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    torch.cuda.empty_cache()

"""# **Inference on test set**"""

model_path = "pbhaskar/meme-caption-generator"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_path, torch_dtype=DTYPE, device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)

# Load the test dataset
with open("./meme_caption_dataset/memes-test.json", "r") as f:
    test_data = json.load(f)

# Function to generate captions
def generate_caption(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")
    prompt = "Caption the given image."

    images = [image]
    prompts = ["<image>" + prompt]
    tokens = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding="longest"
    )
    # print(model.device)
    inputs = tokens.to(DTYPE).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    # Generate caption
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=50)
        output_ids = output_ids[0][input_len:]

    # Decode the generated caption
    caption = processor.batch_decode([output_ids], skip_special_tokens=True)[0]
    print(caption)
    return caption

# Run inference on test dataset
img_dir = "./meme_caption_dataset/memes"  # Update with your actual image directory
predictions = []
references = []

for item in tqdm(test_data):
    img_path = os.path.join(img_dir, item["img_fname"])

    # If image not found, download it from URL (if available)
    if not os.path.exists(img_path):
        image_url = item.get("url", None)
        if image_url:
            try:
                response = requests.get(image_url, stream=True)
                if response.status_code == 200:
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    with open(img_path, "wb") as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                else:
                    print(f"Failed to download: {image_url}")
            except Exception as e:
                print(f"Error downloading image: {e}")

    if not os.path.exists(img_path):
        continue

    # Generate caption
    predicted_caption = generate_caption(model, processor, img_path)
    predictions.append(predicted_caption)

    # Get reference captions
    caption = item["meme_captions"][0] if item["meme_captions"] else ""
    references.append(caption)

# Save generated captions and references to JSON
output_json_path = "./inference_results_caption_generator.json"

results = []
for item, pred, ref in zip(test_data, predictions, references):
    results.append({
        "img_fname": item["img_fname"],
        "generated_caption": pred,
        "reference_caption": ref,
        "url": item.get("url", None)
    })

with open(output_json_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_json_path}")

# 1. BLEU-4 Score
def calculate_bleu4(predictions, references):
    # Tokenize predictions and references
    tokenized_preds = [pred.split() for pred in predictions]
    tokenized_refs = [[ref.split()] for ref in references]  # BLEU expects list of list of references

    # Calculate BLEU-4 score
    smoothie = SmoothingFunction().method1
    bleu4 = corpus_bleu(tokenized_refs, tokenized_preds,
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smoothie)
    return bleu4

# 2. ROUGE-L Score
rouge = evaluate.load('rouge')
def calculate_rouge(predictions, references):
    results = rouge.compute(predictions=predictions, references=references)
    return {
        'rouge1': results['rouge1'],
        'rouge2': results['rouge2'],
        'rougeL': results['rougeL'],
    }

# 3. BERT-F1 Score
bertscore = evaluate.load('bertscore')
def calculate_bert_f1(predictions, references):
    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    return {
        'precision': np.mean(results['precision']),
        'recall': np.mean(results['recall']),
        'f1': np.mean(results['f1'])
    }

# Calculate all metrics
bleu4_score = calculate_bleu4(predictions, references)
rouge_scores = calculate_rouge(predictions, references)
bert_f1_scores = calculate_bert_f1(predictions, references)

# Print results
print(f"BLEU-4 Score: {bleu4_score:.4f}")
print(f"ROUGE-L Score: {rouge_scores['rougeL']:.4f}")
print(f"BERT-F1 Score: {bert_f1_scores['f1']:.4f}")
print("\nDetailed ROUGE Scores:")
for k, v in rouge_scores.items():
    print(f"  {k}: {v:.4f}")
print("\nDetailed BERT Scores:")
for k, v in bert_f1_scores.items():
    print(f"  {k}: {v:.4f}")
