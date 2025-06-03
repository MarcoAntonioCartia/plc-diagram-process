# https://huggingface.co/docs/transformers/model_doc/layoutlmv3
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config

# load pretrained model & processor
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=5)
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")

# Get configuration and build paths
config = get_config()
processed_data_path = Path(config.config['paths']['processed_data'])

# load your custom dataset (assumes HF Dataset JSON)  
dataset = load_dataset("json", data_files={
    "train": str(processed_data_path / "struct" / "train.json"),
    "validation": str(processed_data_path / "struct" / "val.json")
})

def tokenize_and_align(examples):
    encoding = processor(
        images=[img for img in examples["image"]], 
        annotations=examples["annotations"], 
        return_tensors="pt"
    )
    # align labels...
    encoding["labels"] = examples["labels"]
    return encoding

tokenized = dataset.map(tokenize_and_align, batched=True, remove_columns=dataset["train"].column_names)

args = TrainingArguments(
    output_dir="models/layoutlm",
    per_device_train_batch_size=4,
    num_train_epochs=10,
    logging_steps=50,
    save_steps=500
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=processor
)
trainer.train()
