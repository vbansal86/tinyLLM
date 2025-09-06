import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformernew import MiniGPT
from torch import nn
from dataloaders import get_data_splits
from torch.utils.data import DataLoader
from utils import collate_fn_with_padding
import torch 


class TrainingConfig:
    # Model parameters
    vocab_size = 32000
    d_model = 512
    num_heads = 8
    num_layers = 6
    max_seq_len = 1024
    
    # Training parameters
    batch_size = 8  # Adjust based on GPU memory
    learning_rate = 5e-4
    weight_decay = 0.01
    max_epochs = 10
    warmup_steps = 1000
    gradient_clip_val = 1.0

if __name__ == "__main__":
    config = TrainingConfig()
    
    # Initialize model
    model = MiniGPT(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len
    )
    
    # Setup training components
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop


    train_ds, eval_ds, tokenizer = get_data_splits()
    batch_size = 8  # tune to your GPU memory

    train_dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_padding,
        num_workers=4,
        pin_memory=True
    )

    eval_dataloader = None
    if eval_ds is not None:
        eval_dataloader = DataLoader(
            eval_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_with_padding,
            num_workers=4,
            pin_memory=True
        )

    model.train()
    for epoch in range(config.max_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids']
            labels = input_ids[:, 1:].contiguous()  # Shift for next-token prediction
            input_ids = input_ids[:, :-1].contiguous()
            
            # Forward pass
            logits = model(input_ids)
            loss = criterion(logits.view(-1, config.vocab_size), labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        scheduler.step()
        print(f"Epoch {epoch} completed. Average loss: {total_loss/len(train_dataloader):.4f}")
