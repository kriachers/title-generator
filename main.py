"""
FOR TERMINAL:
pip install transformers
pip install datasets
pip install sentencepiece
pip install evaluate
pip install rouge_score
pip install evaluate
pip install torch
"""

# default libraries (no need extra import)
import pandas as pd
import numpy as np

# other libraries
import transformers
import datasets
import sentencepiece
import torch

# libs for evaluation
import evaluate
import rouge_score
rouge = evaluate.load("rouge")

from datasets import Dataset

# Import modules from transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-base")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="t5-base")

"""
CHANGE PATH IF NEEDED 
"""

file_path = '/Users/liza/PycharmProjects/title-generator/title_generation_big.csv'

"""
FILE READING
TEST/TRAIN SPLITTING
"""

title_df = pd.read_csv(file_path)
title_dataset = Dataset.from_pandas(title_df)
title_dataset = title_dataset.train_test_split(test_size=0.2)


"""
TEXT FOR MODEL PREPROCESSING:
TOKENIZING with T5Tokenizer
"""

def preprocess_function(item):
   inputs = ["summarize: " + doc for doc in item["text"]]
   model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
   labels = tokenizer(text_target=item["clickbate_title"], max_length=15, truncation=True)
   model_inputs["labels"] = labels["input_ids"]
   return model_inputs


tokenized_title_dataset = title_dataset.map(preprocess_function, batched=True)

"""
FOR EVALUATION
"""

def compute_metrics(eval_pred):
   # Unpacks the evaluation predictions tuple into predictions and labels.
   predictions, labels = eval_pred


   # Decodes the tokenized predictions back to text, skipping any special tokens (e.g., padding tokens).
   decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)


   # Replaces any -100 values in labels with the tokenizer's pad_token_id.
   # This is done because -100 is often used to ignore certain tokens when calculating the loss during training.
   labels = np.where(labels != -100, labels, tokenizer.pad_token_id)


   # Decodes the tokenized labels back to text, skipping any special tokens (e.g., padding tokens).
   decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)


   # Computes the ROUGE metric between the decoded predictions and decoded labels.
   # The use_stemmer parameter enables stemming, which reduces words to their root form before comparison.
   result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)


   # Calculates the length of each prediction by counting the non-padding tokens.
   prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]


   # Computes the mean length of the predictions and adds it to the result dictionary under the key "gen_len".
   result["gen_len"] = np.mean(prediction_lens)


   # Rounds each value in the result dictionary to 4 decimal places for cleaner output, and returns the result.
   return {k: round(v, 4) for k, v in result.items()}


"""
PARAMETERS FOR MODEL
"""


model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

training_args = Seq2SeqTrainingArguments(
   output_dir="my_fine_tuned_t5-base_model",
   evaluation_strategy="epoch",
   learning_rate=2e-5,
   per_device_train_batch_size=3,
   per_device_eval_batch_size=3,
   weight_decay=0.01,
   save_total_limit=3,
   num_train_epochs=4,
   predict_with_generate=True,
   fp16=False,  # Disable mixed precision training
)

"""
MODEL TRAIN
"""

trainer = Seq2SeqTrainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_title_dataset["train"],
   eval_dataset=tokenized_title_dataset["test"],
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)

trainer.train()

"""
MODEL SAVE
"""

model_path = "clickbate_title_new_t5-base_model"
model = T5ForConditionalGeneration.from_pretrained(model_path)

"""
MODEL TEST ON TEXT EXAMPLE
"""

# Define the input text (from test set)
text = title_dataset['test'][4]['text']
text = "summarize: " + text


# Tokenize and generate the summary
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_length=15, min_length=5)


# Decode and print the summary
pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(pred)
