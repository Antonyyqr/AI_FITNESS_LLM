#!/usr/bin/env python3
# train_simple.py

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

def main():
    # 1️⃣ Load your workout‐only JSONL instead of the full meal+workout data
    dataset = load_dataset(
        "json",
        data_files="fitness_workout_only.jsonl",
        split="train"
    )

    # 2️⃣ Choose a lightweight model for CPU fine‐tuning
    model_name = "distilgpt2"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForCausalLM.from_pretrained(model_name)

    # 3️⃣ Ensure there’s a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # 4️⃣ Tokenize each example, concatenating prompt+completion
    def tokenize_fn(ex):
        text = ex["prompt"] + "\n" + ex["completion"]
        toks = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128
        )
        toks["labels"] = toks["input_ids"].copy()
        return toks

    tokenized = dataset.map(tokenize_fn, batched=False)
    tokenized = tokenized.remove_columns(["prompt", "completion"])

    # 5️⃣ Prepare collator for causal LM (no masked‐language‐modeling)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 6️⃣ Set up training arguments for CPU
    args = TrainingArguments(
        output_dir="ft_model_simple",    # will contain your fine‐tuned weights
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        logging_steps=20,
        save_steps=50,
        save_total_limit=2,
        no_cuda=True,                    # force CPU even if GPU/MPS present
        report_to="none"
    )

    # 7️⃣ Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator
    )

    # 8️⃣ Run training and save
    trainer.train()
    trainer.save_model("ft_model_simple")

if __name__ == "__main__":
    main()
