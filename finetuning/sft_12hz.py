# coding=utf-8
import argparse
import json
import os
import shutil
import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file, load_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    args = parser.parse_args()

    # 强制将说话人名称转为小写，避免推理时的匹配地狱
    FINAL_SPK_NAME = args.speaker_name.strip().lower()

    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")

    MODEL_PATH = args.init_model_path
    if not os.path.exists(MODEL_PATH):
        try:
            from huggingface_hub import snapshot_download
            MODEL_PATH = snapshot_download(repo_id=MODEL_PATH, local_files_only=True)
        except Exception:
            raise FileNotFoundError(f"本地找不到模型: {args.init_model_path}")

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    # 锁定不必要的层，确保稳定性
    if qwen3tts.model.speaker_encoder:
        qwen3tts.model.speaker_encoder.eval()
        for p in qwen3tts.model.speaker_encoder.parameters(): p.requires_grad = False
    
    if hasattr(qwen3tts.model.talker.model, "text_embedding"):
        for p in qwen3tts.model.talker.model.text_embedding.parameters(): p.requires_grad = False

    train_data = [json.loads(line) for line in open(args.train_jsonl).readlines()]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(filter(lambda p: p.requires_grad, qwen3tts.model.parameters()), lr=args.lr, weight_decay=0.01)
    model, optimizer, train_dataloader = accelerator.prepare(qwen3tts.model, optimizer, train_dataloader)

    # 提取全局稳健音色
    accelerator.print(f"[Prep] 正在提取 {FINAL_SPK_NAME} 的全局音色特征...")
    target_speaker_embedding = None
    embedding_count = 0
    with torch.no_grad():
        for i, batch in enumerate(train_dataloader):
            if i > 20: break
            ref_mels = batch['ref_mels'].to(model.device).to(model.dtype)
            emb = model.speaker_encoder(ref_mels).detach()
            if target_speaker_embedding is None:
                target_speaker_embedding = emb.mean(dim=0, keepdim=True)
                embedding_count = ref_mels.size(0)
            else:
                target_speaker_embedding = (target_speaker_embedding * embedding_count + emb.sum(dim=0, keepdim=True)) / (embedding_count + ref_mels.size(0))
                embedding_count += ref_mels.size(0)
    
    accelerator.print(f"[Prep] 音色特征提取完成。")

    model.train()
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = target_speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding
                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(inputs_embeds=input_embeddings[:, :-1, :], attention_mask=attention_mask[:, :-1], labels=codec_0_labels[:, 1:], output_hidden_states=True)
                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]
                _, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
                loss = outputs.loss + sub_talker_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients: accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0: accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            # 更新 config.json
            config_file = os.path.join(output_dir, "config.json")
            with open(config_file, 'r', encoding='utf-8') as f: config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            # 强制清空旧映射，仅保留当前说话人，防止冲突
            talker_config["spk_id"] = {FINAL_SPK_NAME: 3000}
            talker_config["spk_is_dialect"] = {FINAL_SPK_NAME: False}
            config_dict["talker_config"] = talker_config
            with open(config_file, 'w', encoding='utf-8') as f: json.dump(config_dict, f, indent=2, ensure_ascii=False)

            # 保存权重，强制覆盖 ID 3000
            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}
            keys_to_drop = [k for k in state_dict.keys() if k.startswith("speaker_encoder")]
            for k in keys_to_drop: del state_dict[k]

            # 关键：确保 Embedding 层被正确修改并保存
            emb_weight = state_dict['talker.model.codec_embedding.weight']
            # 注入新音色到 3000 坑位
            emb_weight[3000] = target_speaker_embedding[0].detach().to("cpu").to(emb_weight.dtype)
            state_dict['talker.model.codec_embedding.weight'] = emb_weight
            
            save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
            accelerator.print(f"[Done] Checkpoint {epoch} 已保存，说话人: {FINAL_SPK_NAME}")

if __name__ == "__main__":
    train()