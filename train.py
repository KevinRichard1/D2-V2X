import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import torch
from transformers import AutoProcessor, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from safetensors.torch import save_file, load_file
from functools import partial

# Custom modules
from models.d2v2x_model import D2V2XModel
from data_pipeline.dataset import D2V2XDataset
from data_pipeline.collator import D2V2XDataCollator

def parse_args():
    '''Parse arguments'''
    parser = argparse.ArgumentParser(description="Train D2-V2X model")
    # Paths
    parser.add_argument('--qwen_path', type=str, default='./qwen')
    parser.add_argument('--train_path', type=str, default='./data/datasets/d2_v2x_train.json')
    parser.add_argument('--val_path', type=str, default='./data/datasets/d2_v2x_val.json')
    parser.add_argument('--img_path', type=str, default='./data/train/images')
    parser.add_argument('--train_feature_path', type=str, default='./data/tumtraf_features/train')
    parser.add_argument('--val_feature_path', type=str, default='./data/tumtraf_features/val')
    parser.add_argument('--output_path', type=str, default='./checkpoints')

    # Config
    parser.add_argument('--mode', type=str, default='v2x') # 'ego', 'v2x', 'image_only'
    parser.add_argument('--stage', type=int, default=1) # 1=MLP-only, 2=MLP and VLM
    parser.add_argument('--mlp_ckpt', type=str, default=None)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)

    # Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--accum_steps', default=16, type=int)

    return parser.parse_args()


def setup_model_and_processor(qwen_path: str, mode: str, stage: int, mlp_ckpt: str = None):
    '''Initialize LoRA model'''
    print(f"Initializing processor from {qwen_path}...")
    processor = AutoProcessor.from_pretrained(qwen_path, local_files_only=True, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"

    print(f"Loading D2V2X Model in 4 bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    d2v2x_model = D2V2XModel(qwen_path, mode, quantization_config=bnb_config)
    if hasattr(d2v2x_model.model, "config"):
        d2v2x_model.model.config.use_cache = False

    if hasattr(d2v2x_model, 'lidar_mlp'):
        d2v2x_model.lidar_mlp.to(torch.bfloat16)

    d2v2x_model.model = prepare_model_for_kbit_training(
        d2v2x_model.model, 
        use_gradient_checkpointing=True
    )

    # Load pretrained MLP checkpoint
    if mlp_ckpt is not None and hasattr(d2v2x_model, 'lidar_mlp'):
        print(f"Loading pre-trained LiDAR MLP weights from {mlp_ckpt}...")
        full_state_dict = load_file(mlp_ckpt)

        mlp_state_dict = {}
        for k, v in full_state_dict.items():
            if k.startswith("lidar_mlp."):
                new_key = k.replace("lidar_mlp.", "", 1)
                mlp_state_dict[new_key] = v
            else:
                mlp_state_dict[k] = v

        if len(mlp_state_dict) == 0:
            print("WARNING: No LiDAR MLP weights found in checkpoint")
        else:
            d2v2x_model.lidar_mlp.load_state_dict(mlp_state_dict, strict=True)
            print("Successfully loaded LiDAR MLP weights.")

    if stage == 2:
        print("Applying LoRA Config...")
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj", 
                "embed_tokens", "lm_head"
            ]
        )
        d2v2x_model.model = get_peft_model(d2v2x_model.model, lora_config)

    # Freeze params
    if stage == 1:
        for name, param in d2v2x_model.named_parameters():
            if "lidar_mlp" in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        for name, param in d2v2x_model.named_parameters():
            if "lidar_mlp" in name.lower() and hasattr(d2v2x_model, 'lidar_mlp'):
                param.requires_grad = True

    trainable_count = sum(p.numel() for p in d2v2x_model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in d2v2x_model.parameters())
    print(f"Trainable Parameters: {trainable_count:,} ({100 * trainable_count / total_count:.2f}% of total)")

    return d2v2x_model, processor

def setup_datasets(args):
    '''Initialize and return datasets'''
    train_dataset = D2V2XDataset(
        args.train_path,
        args.img_path,
        args.train_feature_path,
        mode=args.mode,
        is_training=True
    )

    eval_dataset = D2V2XDataset(
        args.val_path,
        args.img_path,
        args.val_feature_path,
        mode=args.mode,
        is_training=True
    )

    return train_dataset, eval_dataset

def main():
    # Initialization
    args = parse_args()
    print(f"Starting Stage {args.stage} in {args.mode} mode...")
    d2v2x_model, processor = setup_model_and_processor(args.qwen_path, args.mode, args.stage, args.mlp_ckpt)

    train_dataset, eval_dataset = setup_datasets(args)
    print(f"Datasets Loaded: Train={len(train_dataset)}, Val={len(eval_dataset)}")

    data_collator = D2V2XDataCollator(processor)

    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_path,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.05,
        optim="adamw_torch_fused",
        dataloader_num_workers=0,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum_steps,
        eval_accumulation_steps=1,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to=["wandb"],
        logging_steps=10,
        eval_strategy="epoch" if args.stage == 2 else "no",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
    )

    # Initialize trainer
    trainer = Trainer(
        model=d2v2x_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    print("Starting Trainer.train()")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_save_dir = f"{args.output_path}/final_model"
    os.makedirs(final_save_dir, exist_ok=True)

    # Save LiDAR MLP
    if hasattr(d2v2x_model, 'lidar_mlp'):
        mlp_save_path = f"{final_save_dir}/lidar_mlp.safetensors"
        save_file(d2v2x_model.lidar_mlp.state_dict(), mlp_save_path)
        print(f"LiDAR MLP weights saved to {mlp_save_path}")

    if args.stage == 2:
        # Save LoRA adapters
        lora_save_path = f"{final_save_dir}/lora"
        d2v2x_model.model.save_pretrained(lora_save_path)
        processor.save_pretrained(lora_save_path)
        print(f"LoRA adapters and processor saved to {lora_save_path}")

if __name__ == "__main__":
    main()