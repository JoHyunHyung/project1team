from datasets import load_dataset
import os

from dataclasses import dataclass

class TrainingConfig:
    image_size =128
    train_batch_size=16
    eval_batch_size=16
    num_epochs=50
    gradient_accumulation_steps=1
    learning_rate=1e-4
    lr_warmup_step=500
    save_image_epochs=10
    save_model_epochs=50
    mixed_precision='fp16'
    output_dir='ddpm-butterflies-128'

    push_to_hub=False
    hub_private_repo=False
    overwrite_output_dir=True
    seed=777

config=TrainingConfig()

config.dataset_name="hahminlew/kream-product-blip-captions"
dataset=load_dataset(config.dataset_name,split='train')

os.makedirs("dataset/", exist_ok=True)

dataset_list=list(dataset)

from tqdm import tqdm
for i, example in tqdm(enumerate(dataset_list)):
    image_pil=example["image"] #PIL이미지 객체
    image_path= f"dataset/fashion_{i}.png"  #이미지 파일 경로 설정
    image_pil.save(image_path) #PIL 이미지를 파일로 저장