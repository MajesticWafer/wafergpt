# Tuning

### Step-by-Step Guide to Fine-Tuning a GPT-2 Model on Your Data

1. **Create Your Project Directory**  
   First, create a new directory for your project and navigate into it:  
   `mkdir my-llm-project`
   `cd my-llm-project`

2. **Install Required Libraries**  
   Install the required Python libraries, including `torch`, `transformers`, `datasets`, and `accelerate`:  
   `pip install torch torchvision torchaudio transformers datasets accelerate`

3. **Create Your Custom Data File**  
   Create a text file named `my_data.txt` to hold your training data:  
   `nano my_data.txt`
   Add your text data that reflects how you write and think.

4. **Create the Training Script**  
   Create a Python script named `train.py`:  
   `nano train.py` 
   Insert the following code into `train.py`:  
   ```
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
   ```

5. **Run the Training Script**  
   Execute the training script:  
   `python train.py`

6. **Create the Generation Script**  
   Create another Python script named `generate.py`:  
   `nano generate.py`
   Insert the following code into `generate.py`:  
   ```
   from transformers import GPT2Tokenizer, GPT2LMHeadModel

   # Load the fine-tuned model and tokenizer
   tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_model")
   model = GPT2LMHeadModel.from_pretrained("./fine_tuned_model")

   # Prompt for generating text
   input_text = "What do you think about"
   inputs = tokenizer(input_text, return_tensors="pt")
   outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)

   # Decode and print the generated text
   generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
   print(generated_text)
   ```

7. **Run the Generation Script**  
   Execute the generation script to see your model in action:  
   `python generate.py`

### Conclusion
You have now set up a local LLM that can generate text based on your own data! Feel free to modify your dataset and experiment with the training parameters to improve the model's performance.
