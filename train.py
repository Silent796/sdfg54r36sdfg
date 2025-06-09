import os
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.amp import autocast
from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
from huggingface_hub import HfApi, upload_folder
from lion_pytorch import Lion
import json
import re
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tokenizers import Tokenizer # ADDED

# --- Configuration ---
NUM_EPOCHS = 1
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
LEARNING_RATE = 1e-4
SEQ_LEN = 1024
VALIDATION_SPLIT = 0.1

MODEL_DIM = 768
MODEL_DEPTH = 12
MODEL_HEADS = 12

GENERATE_EVERY = 50
GENERATE_LENGTH = 1024
SAVE_EVERY = 500
CHECKPOINT_DIR = 'checkpoints'
FINAL_MODEL_DIR = 'final_model'

MODEL_NAME = ""
HF_REPO_ID = ""
HF_TOKEN = ""

# --- Tokenizer and Special Tokens ---
TOKENIZER_FILE = "bpe-tokenizer.json" # ADDED: Path to our trained tokenizer

# Load the tokenizer to get special tokens and vocab size
if not os.path.exists(TOKENIZER_FILE):
    raise FileNotFoundError(f"Tokenizer file not found at {TOKENIZER_FILE}. Please run train_tokenizer.py first.")
tokenizer = Tokenizer.from_file(TOKENIZER_FILE) # ADDED

# CHANGED: Get tokens and IDs from the tokenizer
USER_TOKEN_STR = "<|user|>"
ASSISTANT_TOKEN_STR = "<|assistant|>"
END_TOKEN_STR = "<|end|>"
PAD_TOKEN_ID = tokenizer.token_to_id("<|pad|>")
VOCAB_SIZE = tokenizer.get_vocab_size()

def setup_distributed(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    dist.destroy_process_group()

# REMOVED: decode_bytes function is no longer needed. We use tokenizer.decode()

def calculate_perplexity(loss):
   return math.exp(loss) if isinstance(loss, float) else torch.exp(loss).item()

def create_model():
   model = TransformerWrapper(
       num_tokens=VOCAB_SIZE, # CHANGED: Use the BPE vocab size
       max_seq_len=SEQ_LEN,
       num_memory_tokens=40,
       attn_layers=Decoder(
           dim=MODEL_DIM, depth=MODEL_DEPTH, heads=MODEL_HEADS,
           ff_no_bias=True, attn_flash=True, rotary_xpos=True, ff_glu=True, shift_tokens=1,
           attn_qk_norm=True, attn_qk_norm_dim_scale=True, attn_qk_norm_scale=10,
           ff_dropout=0.1, attn_dropout=0.1, use_dynamic_tanh=True, dynamic_tanh_init_alpha=1.5,
           num_residual_streams=1, integrate_layers=False
       )
   )
   return AutoregressiveWrapper(model, pad_value=PAD_TOKEN_ID) # CHANGED: Inform the wrapper about the pad token ID

# CHANGED: The entire Dataset class is updated to use the BPE tokenizer
class JsonlConversationDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>")
        if self.pad_token_id is None:
            # Fallback if pad token isn't in the tokenizer for some reason
            self.pad_token_id = 0
            print("Warning: '<|pad|>' token not found. Using 0 for padding.")
        self.chunks = self._load_and_process_data(file_path)

    def _load_and_process_data(self, file_path):
        print(f"Process {os.environ.get('RANK', 0)}: Loading and processing data from {file_path}...")
        all_chunks = []
        # Add errors='ignore' for robustness during file reading
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    conversation_turns = json.loads(line)
                    full_conversation_str = ""
                    for turn in conversation_turns:
                        role, content = turn.get('role'), turn.get('content', '')

                        # --- THIS IS THE CRUCIAL FIX ---
                        # Clean the string to remove invalid surrogate characters, just like in the tokenizer trainer
                        if content:
                            content = content.encode('utf-8', 'surrogatepass').decode('utf-8', 'replace')
                        # -----------------------------

                        if role == 'user': full_conversation_str += USER_TOKEN_STR + content + END_TOKEN_STR
                        elif role == 'assistant': full_conversation_str += ASSISTANT_TOKEN_STR + content + END_TOKEN_STR

                    # Use the BPE tokenizer to encode the string into token IDs
                    token_ids = self.tokenizer.encode(full_conversation_str).ids

                    chunk_size = self.seq_len + 1
                    for i in range(0, len(token_ids), chunk_size):
                        chunk = token_ids[i:i + chunk_size]
                        if len(chunk) < chunk_size:
                            chunk.extend([self.pad_token_id] * (chunk_size - len(chunk)))
                        all_chunks.append(torch.tensor(chunk).long())
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line in process {os.environ.get('RANK', 0)}")
                    continue

        print(f"Process {os.environ.get('RANK', 0)}: Finished processing. Found {len(all_chunks)} chunks.")
        return all_chunks

    def __len__(self): return len(self.chunks)
    def __getitem__(self, idx): return self.chunks[idx]

# CHANGED: Generation now uses the BPE tokenizer for encoding and decoding
def generate_sample(model, rank, length=GENERATE_LENGTH):
    model.eval()
    start_text = USER_TOKEN_STR + "I have leg pain." + END_TOKEN_STR + ASSISTANT_TOKEN_STR
    
    # Encode the prompt using the tokenizer
    input_ids = tokenizer.encode(start_text).ids
    input_tensor = torch.tensor([input_ids]).to(rank).long()

    with autocast(device_type='cuda', dtype=torch.bfloat16):
        # The model generates token IDs
        generated_ids = model.generate(input_tensor, length, eos_token=tokenizer.token_to_id(END_TOKEN_STR))

    # Decode the full sequence of IDs back to a string
    full_sequence_ids = generated_ids[0].cpu().tolist()
    full_generated_text = tokenizer.decode(full_sequence_ids, skip_special_tokens=False)

    # Extract just the response part
    response_only = full_generated_text[len(start_text):].split(END_TOKEN_STR)[0]

    return f"--- PROMPT ---\n{start_text}\n--- RESPONSE ---\n{response_only}"

def save_checkpoint(model, optimizer, global_step, epoch, batch_idx, save_path=CHECKPOINT_DIR):
    os.makedirs(save_path, exist_ok=True)
    model_to_save = model.module if isinstance(model, DDP) else model
    checkpoint = {
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'epoch': epoch,
        'batch_idx': batch_idx
    }
    checkpoint_path = os.path.join(save_path, f'checkpoint_step_{global_step}.pt')
    torch.save(checkpoint, checkpoint_path)
    checkpoints = sorted([os.path.join(save_path, f) for f in os.listdir(save_path)], key=os.path.getmtime)
    if len(checkpoints) > 3: os.remove(checkpoints[0])
    print(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path

# CHANGED: Save the tokenizer config along with the final model
def save_final_model(model, save_dir=FINAL_MODEL_DIR):
    os.makedirs(save_dir, exist_ok=True)
    model_to_save = model.module if isinstance(model, DDP) else model
    model_path = os.path.join(save_dir, 'model.pt')
    torch.save({
        'model_state_dict': model_to_save.state_dict(),
        'model_config': {
            'num_tokens': VOCAB_SIZE, 'max_seq_len': SEQ_LEN, 'dim': MODEL_DIM,
            'depth': MODEL_DEPTH, 'heads': MODEL_HEADS
        }
    }, model_path)
    
    # ADDED: Save the tokenizer file with the model for easy reloading/sharing
    tokenizer.save(os.path.join(save_dir, 'tokenizer.json'))
    
    print(f"Final model and tokenizer saved to {save_dir}")
    return save_dir

# ... (train function remains the same) ...
def train(model, train_loader, val_loader, optimizer, rank, is_ddp, resume_from=None):
    total_batches = len(train_loader)
    global_step = 0
    start_epoch = 0
    
    if resume_from and os.path.exists(resume_from):
        if rank == 0:
            print(f"Loading checkpoint: {resume_from}")
        map_location = {'cuda:0': f'cuda:{rank}'}
        checkpoint = torch.load(resume_from, map_location=map_location)
        
        model_to_load = model.module if is_ddp else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']
        start_epoch = checkpoint['epoch']
        if rank == 0:
            print(f"Resuming from epoch {start_epoch + 1}, step {global_step}")
   
    model.train()
    optimizer.zero_grad()
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        if is_ddp:
            train_loader.sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        accumulated_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(rank)
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = model(batch)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            loss.backward()
            accumulated_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                global_step += 1
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                optimizer.step()
                optimizer.zero_grad()

                if rank == 0:
                    train_loss = accumulated_loss / GRADIENT_ACCUMULATION_STEPS
                    train_perplexity = calculate_perplexity(train_loss)
                    accumulated_loss = 0.0

                    model.eval()
                    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.bfloat16):
                        val_batch = next(iter(val_loader))
                        val_batch = val_batch.to(rank)
                        val_loss = model(val_batch).item()
                    model.train()

                    print(f"Batch {batch_idx}/{total_batches} | Step: {global_step} | Train Loss: {train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | PPL: {train_perplexity:.2f} | "
                          f"Grad: {grad_norm:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

                    if global_step % SAVE_EVERY == 0:
                        save_checkpoint(model, optimizer, global_step, epoch, batch_idx)

                    if global_step % GENERATE_EVERY == 0:
                        underlying_model = model.module if is_ddp else model
                        print(f"\n{'=' * 20} Sample Generation {'=' * 20}\n{generate_sample(underlying_model, rank)}\n{'=' * 62}\n")
       
    return global_step, total_batches

def upload_to_huggingface(model_dir, repo_id=HF_REPO_ID, token=HF_TOKEN):
    if not repo_id or "YourUsername" in repo_id:
        print("Skipping Hugging Face upload: HF_REPO_ID not set.")
        return
    print(f"Uploading model to Hugging Face Hub: {repo_id}")
    api = HfApi()
    api.upload_folder(folder_path=model_dir, repo_id=repo_id, repo_type="model", token=token, commit_message=f"Upload final model: {MODEL_NAME}")
    print("Upload complete.")

def main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1

    if is_ddp:
        setup_distributed(rank, world_size)

    data_path = '/kaggle/input/transformer-test/train3.jsonl'
    if not os.path.exists(data_path):
        if rank == 0:
            print(f"Error: Data file not found at '{data_path}'. Please create it first.")
        return

    model = create_model()
    model = model.to(rank)

    if rank == 0:
        print(f"Model created with BPE tokenizer. Vocabulary size: {VOCAB_SIZE}") # CHANGED
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Training with micro-batch size: {BATCH_SIZE}")
        print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
        print(f"Effective batch size per step: {EFFECTIVE_BATCH_SIZE * world_size}")

    if is_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # CHANGED: Pass the tokenizer object to the dataset
    full_dataset = JsonlConversationDataset(data_path, tokenizer, SEQ_LEN)
    if len(full_dataset) == 0:
        if rank == 0:
            print("Error: The dataset is empty. Check the data file and processing logic.")
        return

    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, shuffle=(train_sampler is None))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, shuffle=False)

    if rank == 0:
        print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    optimizer = Lion(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)

    resume_from = ''

    global_step, total_batches = train(model, train_loader, val_loader, optimizer, rank, is_ddp, resume_from=resume_from)

    if rank == 0:
        save_checkpoint(model, optimizer, global_step, NUM_EPOCHS, total_batches)
        model_dir = save_final_model(model)
        upload_to_huggingface(model_dir)

    if is_ddp:
        cleanup_distributed()

if __name__ == "__main__":
    main()
