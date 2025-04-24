# #!/usr/bin/env python3
# # train.py

# from datasets import load_dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     Trainer,
#     TrainingArguments,
#     DataCollatorForLanguageModeling
# )

# def main():
#     # 1) Load your JSONL dataset
#     data = load_dataset("json", data_files="fitness_data.jsonl", split="train")

#     # 2) Choose a small base model
#     model_name = "distilgpt2"  # or "gpt2" if you prefer
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model     = AutoModelForCausalLM.from_pretrained(model_name)

#     # 3) Tokenize with padding + labels
#     def tokenize_fn(example):
#         # concatenate prompt + completion, truncate & pad to max_length
#         tokens = tokenizer(
#             example["prompt"] + example["completion"],
#             padding="max_length",
#             truncation=True,
#             max_length=128
#         )
#         tokens["labels"] = tokens["input_ids"].copy()
#         return tokens

#     tokenized = data.map(tokenize_fn, batched=False)

#     # 4) Set up a data collator that will pad batches dynamically
#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer,
#         mlm=False            # causal language modeling
#     )

#     # 5) Define training arguments
#     training_args = TrainingArguments(
#         output_dir="ft_model",
#         overwrite_output_dir=True,
#         num_train_epochs=3,
#         per_device_train_batch_size=2,
#         logging_steps=10,
#         save_steps=50,
#         save_total_limit=2,
#         report_to="none"    # disable wandb/other loggers
#     )

#     # 6) Initialize Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized,
#         data_collator=data_collator
#     )

#     # 7) Train & save
#     trainer.train()
#     trainer.save_model("ft_model")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# train.py

#!/usr/bin/env python3
# train.py

#!/usr/bin/env python3
# train.py

#!/usr/bin/env python3
# train.py

#!/usr/bin/env python3
# train.py

import os
# ⚠️ MUST be set *before* any MPS work to allow dropout fallback on CPU
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch._dynamo
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# Suppress any Dynamo compile errors by falling back to eager
torch._dynamo.config.suppress_errors = True

def main():
    # 1) Load your generated JSONL dataset
    data = load_dataset("json",
                        data_files="fitness_data_big.jsonl",
                        split="train")

    # 2) Pick a powerful base model
    model_name = "gpt2-large"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16   # load weights in half-precision
    )

    # 3) Ensure pad token is defined
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # 4) Tokenize & create labels
    def tokenize_fn(example):
        text = example["prompt"] + "\n" + example["completion"]
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = data.map(tokenize_fn, batched=False)
    tokenized = tokenized.remove_columns(["prompt", "completion"])

    # 5) Causal-LM data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 6) TrainingArguments
    training_args = TrainingArguments(
        output_dir="ft_model_best",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        logging_steps=50,
        save_steps=200,
        save_total_limit=3,
        fp16=False,                  # must be False on MPS
        remove_unused_columns=False,
        report_to="none"
    )

    # 7) Memory savings
    model.gradient_checkpointing_enable()

    # 8) Compile with AOT-eager (avoids MPS-Inductor welford & dropout issues)
    model = torch.compile(model, backend="aot_eager")

    # 9) Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator
    )

    # 🔟 Train & save
    trainer.train()
    trainer.save_model("ft_model_best")

if __name__ == "__main__":
    main()
