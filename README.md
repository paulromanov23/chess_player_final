# Chess AI — 135M-Parameter Transformer + LoRA Fine-tune

A transformer trained from scratch to play chess using reinforcement learning self-play, 
plus a LoRA fine-tune of a 0.5B Qwen model — all built in PyTorch without wrappers 
around someone else's code.

## How it works
- Custom transformer architecture designed for chess position encoding
- RL self-play training loop with custom reward shaping
- Separate LoRA/PEFT fine-tune of Qwen-0.5B on chess data
- Full GPU-accelerated training pipeline

## Results
Published on Hugging Face: [parom23/chess_transformer](https://huggingface.co/parom23/chess_transformer)

## Tech stack
PyTorch · Hugging Face · LoRA/PEFT · Qwen · Python
