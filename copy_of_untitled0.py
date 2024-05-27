
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

common_voice["train"] = load_dataset("mozilla-foundation/common_voice_17_0", "hi", split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_17_0", "hi", split="test", use_auth_token=True)

dataset = load_dataset("mozilla-foundation/common_voice_17_0", "hi", split="train", cache_dir="/root/.cache/huggingface/datasets/", trust_remote_code=True)

common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", language="Hindi", task="transcribe")

input_str = common_voice["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3", language="Hindi", task="transcribe")

from datasets import Audio

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    try:
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch
    except Exception as e:
        print(f"Error processing batch: {e}")
        return {
            "audio": None,
            "input_features": None,
            "labels": None
        }

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"])

from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

model.generation_config.language = "hindi"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
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

import evaluate

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids


    label_ids[label_ids == -100] = tokenizer.pad_token_id

 
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-Large-v3-hindi",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=2500,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=False,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

from transformers import Seq2SeqTrainer
from accelerate import Accelerator

# Initialize the accelerator
accelerator = Accelerator()
device = accelerator.device
model.to(device)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)

trainer.train()



kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_17_0",
    "dataset": "Common Voice 17.0",  # a 'pretty' name for the training dataset
    "dataset_args": "config: hi, split: test",
    "language": "hi",
    "model_name": "Whisper Large v3 Trained on Hindi",  # a 'pretty' name for your model
    "finetuned_from": "openai/whisper-large-v3",
    "tasks": "automatic-speech-recognition",
}
trainer.push_to_hub(**kwargs)
