# -*- coding: utf-8 -*-
"""
Refactored script from Jupyter Notebook for standalone execution
"""

# Install necessary packages if not already installed
# !pip install --upgrade datasets[audio] transformers accelerate evaluate jiwer tensorboard gradio

from huggingface_hub import login
from datasets import load_dataset, DatasetDict
from transformers import (WhisperFeatureExtractor, WhisperTokenizer, 
                          WhisperProcessor, WhisperForConditionalGeneration,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer)
from datasets import Audio
import evaluate

# Login to Hugging Face (comment out if not needed)
# login()

# Load dataset
common_voice = DatasetDict()
common_voice["train"] = load_dataset("mozilla-foundation/common_voice_17_0", "hi", split="train+validation").select(range(100))
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_17_0", "hi", split="test").select(range(100))

print(common_voice)

# Remove unnecessary columns
common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

# Initialize feature extractor and tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", language="Hindi", task="transcribe")

# Tokenization example
input_str = common_voice["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")

# Initialize processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3", language="Hindi", task="transcribe", n_proc = 4)

print(common_voice["train"][0])

# Cast audio column
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

print(common_voice["train"][0])

# Prepare dataset function
def prepare_dataset(batch):
    try:
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["input_length"] = len(audio["array"]) // processor.feature_extractor.sampling_rate
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    except Exception as e:
        print(f"Error in processing batch: {e}")
    return batch

# Apply dataset preparation
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice["train"].column_names)

# Load model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

# Data collator
class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor, decoder_start_token_id):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Evaluation metric
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-large-v3-hi",
    per_device_train_batch_size=16,
    learning_rate=1e-5,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    logging_dir='./logs',
    logging_steps=500,
    save_steps=1000,
    eval_steps=1000,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True
)

# Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()
