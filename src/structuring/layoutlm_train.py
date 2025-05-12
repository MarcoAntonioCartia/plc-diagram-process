# https://huggingface.co/docs/transformers/model_doc/layoutlmv3
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# load pretrained model & processor:contentReference[oaicite:4]{index=4}
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=5)
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")

# load your custom dataset (assumes HF Dataset JSON)  
dataset = load_dataset("json", data_files={
    "train": "../../data/struct/train.json",
    "validation": "../../data/struct/val.json"
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