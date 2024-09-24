from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset from text file
def load_custom_dataset(file_path):
    dataset = load_dataset('text', data_files={'train': file_path})
    return dataset['train']

# Load the tokenizer and model
model_name = "gpt2"  # You can use 'gpt-neo', 'gpt2-medium', etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Load and tokenize dataset
dataset = load_custom_dataset("my_data.txt")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    per_device_train_batch_size=2,  # Adjust based on your GPU capacity
    per_device_eval_batch_size=2,
    num_train_epochs=3,  # Number of epochs to train
    save_steps=500,  # Save the model every 500 steps
    logging_steps=100,  # Log every 100 steps
    fp16=True,  # Mixed precision training (for RTX 3060)
    save_total_limit=2,  # Only keep last 2 checkpoints
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start training
trainer.train()

# Save the final model
trainer.save_model("./fine_tuned_model")
