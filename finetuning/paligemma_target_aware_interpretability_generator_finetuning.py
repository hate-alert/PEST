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
import pandas as pd

# Define global DTYPE for consistent usage
DTYPE = torch.bfloat16

import pandas as pd

class HateExplainDataset(Dataset):
    def __init__(self, jsonl_file, csv_file, img_dir, split="train"):
        # Load hateful memes from JSONL file
        self.hateful_data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                item['img'] = "img/" + item['img']
                # Add the explanation field if not present
                if "explanation" not in item and "reasonings" in item:
                    item["explanation"] = item["reasonings"][0]
                self.hateful_data.append(item)

        # Load not-hateful memes from CSV file
        self.not_hateful_df = pd.read_csv(csv_file)

        # Combine both datasets
        self.data = []

        # Add hateful memes
        for item in self.hateful_data:
            self.data.append({
                "id": item["id"],
                "img": item["img"],
                "label": 1,
                "explanation": item.get("explanation", "")
            })

        # Add not-hateful memes
        wrong_labeled_count=0
        for _, row in self.not_hateful_df.iterrows():
            if 'verification' in self.not_hateful_df.columns:
                if row['verification'] == "-1" or row['explanation'] == "-1":
                    wrong_labeled_count+=1
                    continue
            self.data.append({
                "id": str(row["id"]),
                "img": row["img"],
                "label": int(row["label"]),
                "explanation": row["explanation"]
            })

        print("Count of wrong labeled data points: ", wrong_labeled_count)

        # Split data (80% train, 20% validation)
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
        img_path = os.path.join(self.img_dir, item["img"])
        image = Image.open(img_path).convert("RGB")

        # Get label and explanation
        label = item["label"]
        explanation = item["explanation"]

        # Create prompt based on label
        prompt = f"Explain why this meme is {'hateful' if label == 1 else 'not-hateful'}"  # target aware

        return (image, prompt, explanation)

# from transformers import BitsAndBytesConfig
# from peft import get_peft_model, LoraConfig

# # Configure quantization
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# # Configure LoRA
# lora_config = LoraConfig(
#     r=8,
#     target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
#     task_type="CAUSAL_LM",
# )

def train_hate_explain_model(jsonl_file, csv_file, img_dir, output_dir):
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

    # model = PaliGemmaForConditionalGeneration.from_pretrained(
    #     model_name,
    #     device_map="auto",
    #     quantization_config=bnb_config
    # )

    # # Apply LoRA
    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()

    # Create datasets
    train_dataset = HateExplainDataset(jsonl_file, csv_file, img_dir, split="train")
    eval_dataset = HateExplainDataset(jsonl_file, csv_file, img_dir, split="val")
    
    for image, text, label in train_dataset:
        print(image, text, label)
        break

    # Define collate function
    def collate_fn(examples):
        images = [ex[0] for ex in examples]
        prompts = ["<image>" + ex[1] for ex in examples]
        explanations = [ex[2] for ex in examples]

        tokens = processor(
            text=prompts,
            images=images,
            suffix=explanations,
            return_tensors="pt",
            padding="longest"
        )

        # tokens = tokens.to(DTYPE).to(model.device)
        return tokens

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        eval_steps=1500,
        logging_dir=f"{output_dir}/logs",
        logging_steps=200,
        save_steps=1000,
        save_total_limit=2,
        warmup_steps=2,
        learning_rate=2e-5,
        weight_decay=1e-6,
        adam_beta2=0.999,
        fp16=False,
        bf16=True,
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

    return model, processor

if __name__ == "__main__":
    # Train model
    model, processor = train_hate_explain_model(
        "FHM_train_hateful_explanations.jsonl",
        "FHM_train_not-hateful_explanations.csv",
        "hateful_memes",
        "meme-explain-model"
    )

    # Save model and processor
    model.save_pretrained("meme-explain-model/final")
    processor.save_pretrained("meme-explain-model/final")

    # Push to Hugging Face Hub
    model.push_to_hub("pbhaskar/meme-explanation-generator")
    processor.push_to_hub("pbhaskar/meme-explanation-generator")

# Clear GPU Memory
import gc

gc.collect()
torch.cuda.empty_cache()

for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    torch.cuda.empty_cache()

"""# **Inference on test set**"""

model_path = "pbhaskar/meme-explanation-generator"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_path, torch_dtype=DTYPE, device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)

# Load the test dataset
test_data = pd.read_csv("HARM_P_test_explanations.csv")

# Function to generate explanations
def generate_explanation(model, processor, image_path, label):
    image = Image.open(image_path).convert("RGB")
    prompt = f"Explain why this meme is {'harmful' if label == 1 else 'not-harmful'}"

    images = [image]
    prompts = ["<image>" + prompt]
    tokens = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding="longest"
    )

    inputs = tokens.to(DTYPE).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    # Generate explanation
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=50)
        output_ids = output_ids[0][input_len:]

    # Decode the generated explanation
    explanation = processor.batch_decode([output_ids], skip_special_tokens=True)[0].strip()
    print("explanation: " ,explanation)
    return explanation

# Run inference on test dataset
img_dir = "harm_p_images"  # Update with your actual image directory
predictions = []
references = []

refined_rows = []  # Will store rows where explanation != -1

wrong_labeled_count=0
for _, item in tqdm(test_data.iterrows()):
    if item['explanation'] == "-1":
        wrong_labeled_count+=1
        continue

    img_path = os.path.join(img_dir, item["img"])


    # label_text = str(item["label"]).lower()
    # if "not harmful" in label_text or "not-harmful" in label_text:
    #     item["label"] = 0
    # else:
    #     item["label"] = 1
    

    # Generate explanation
    predicted_explanation = generate_explanation(model, processor, img_path, item["label"])
    predictions.append(predicted_explanation)

    # Get reference explanation
    reference_explanation = item["explanation"]
    references.append(reference_explanation)

    refined_rows.append(item)  # Keep only valid rows

print("Count of wrong labeled data points: ", wrong_labeled_count)
print("Length of refined_rows: ", len(refined_rows))
print("Length of predictions: ", len(predictions))
print("Length of references: ", len(references))


# Save generated captions and references to JSON
output_json_path = "./inference_results_HARM_P_test_interpretability_generator.json"

results = []
for row, pred, ref in zip(refined_rows, predictions, references):
    results.append({
        "id": row["id"],
        "img": row["img"],
        "label": row["label"],
        "generated_explanation": pred,
        "reference_explanation": ref
    })


with open(output_json_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_json_path}")


# with open(output_json_path, "r") as f:
#     data = json.load(f)

# predictions = [item["generated_explanation"] for item in data]
# references = [item["reference_explanation"] for item in data]


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