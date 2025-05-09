from .model import GPT
from .dataset import NameDataset
from .trainer import Trainer, TrainerConfig

import torch
import random
random.seed(0)

def initialize_vanilla_model(mconf):
    attention_model = None
    ### TODO:
    ### [part d]: Make some model here

    ### START CODE HERE
    attention_model = GPT(mconf)
    ### END CODE HERE
    return attention_model

def initialize_rope_model(mconf, bottleneck_dim=32):
    attention_model = None
    ### TODO
    ### [part h]: Make some other model here

    ### START CODE HERE
    mconf.rope = True
    mconf.pos_encoding_type = 'rope'
    
    # Explicitly set the bottleneck_dim
    mconf.bottleneck_dim = bottleneck_dim
    
    # Initialize the model with RoPE enabled
    attention_model = GPT(mconf)
    
    # Debug print for verification
    print(f"RoPE Model initialized: rope={getattr(attention_model, 'rope', False)}, "
          f"pos_encoding_type={getattr(mconf, 'pos_encoding_type', None)}, "
          f"bottleneck_dim={bottleneck_dim}")
      
    ### END CODE HERE
    return attention_model

def finetune(reading_params_path, finetune_corpus_path, pretrain_dataset, block_size, model, finetune_lr=6e-4, writer=None):
    ### TODO:
    ### [part d] [part f]:
    ### - Given:
    ###     1. A finetuning corpus specified in finetune_corpus_path
    ###     2. A path reading_params_path containing pretrained model
    ###         parameters, or None if finetuning without a pretrained model
    ### - Goals:
    ###     1. If reading_params_path is specified, load these parameters
    ###         into the model
    ###     2. Finetune the model on this corpus
    ###
    ### - Make sure to use the following hyperparameters:
    ###     Hyperparameters for finetuning WITHOUT a pretrained model:
    ###         max_epochs=75
    ###         batch_size=256
    ###         learning_rate=6e-4
    ###         lr_decay=True
    ###         warmup_tokens=512*20
    ###         final_tokens=200*len(pretrain_dataset)*block_size
    ###         num_workers=0
    ###     Hyperparameters for finetuning WITH a pretrained model:
    ###         max_epochs=10
    ###         batch_size=256
    ###         learning_rate=6e-4
    ###         lr_decay=True
    ###         warmup_tokens=512*20
    ###         final_tokens=200*len(pretrain_dataset)*block_size
    ###         num_workers=0
    ###
    ###
    ### Note: Please use torch.load(reading_params_path, map_location=torch.device('cpu'), weights_only=True) to load pretrained model 

    trainer_obj = None #Trainer object (see trainer.py for more details)
    tconf = None #TrainerConfig object (see trainer.py for more details)
    ### START CODE HERE

    # Check if we're loading a pretrained model
    is_pretrained = reading_params_path is not None
    is_rope = hasattr(model, 'rope') and model.rope
    
    # Load the finetuning corpus
    with open(finetune_corpus_path, 'r', encoding='utf-8') as f:
        finetune_corpus = f.read()
    
    # Create the name dataset for finetuning
    name_dataset = NameDataset(finetune_corpus, pretrain_dataset)
    
    if is_pretrained:
        if is_rope:
            # Finetuning WITH a pretrained RoPE model
            max_epochs = 50  # INCREASED from 30 to 50
            actual_lr = finetune_lr  # CHANGED - use standard learning rate
        else:
            # Finetuning WITH a pretrained vanilla model
            max_epochs = 40  # INCREASED from 10 to 40
            actual_lr = finetune_lr
    else:
        # Finetuning WITHOUT a pretrained model
        max_epochs = 75
        actual_lr = finetune_lr
    
    # Common configuration
    tconf = TrainerConfig(
        max_epochs=max_epochs,
        batch_size=256,
        learning_rate=actual_lr,
        lr_decay=True,
        warmup_tokens=512*20,
        final_tokens=200*len(pretrain_dataset)*block_size,
        num_workers=0,
        writer=writer
    )
    
    # Create the trainer object
    trainer_obj = Trainer(model, name_dataset, None, tconf)
    
    # If we have a pretrained model, load its parameters
    if is_pretrained:
        print(f"Loading pretrained model from {reading_params_path}")
        model.load_state_dict(torch.load(reading_params_path, map_location=torch.device('cpu'), weights_only=True))
        print(f"Pretrained model loaded successfully!")
    
    ### END CODE HERE
    return tconf, trainer_obj

def pretrain(pretrain_dataset, block_size, model, pretrain_lr=6e-3, writer=None):
    ### TODO:
    ### [part f]:
    ### - Given:
    ###     1. A corpus specified in pretrain_dataset
    ### - Goals:
    ###     1. Pretrain the model on this corpus
    ###
    ### - Make sure to use the following hyperparameters for pretraining:
    ###     max_epochs=650
    ###     batch_size=128
    ###     learning_rate=6e-3
    ###     lr_decay=True
    ###     warmup_tokens=512*20
    ###     final_tokens=200*len(pretrain_dataset)*block_size
    ###     num_workers=0
    trainer_obj = None #Trainer object (see trainer.py for more details)
    tconf = None #TrainerConfig object (see trainer.py for more details)

    ### START CODE HERE

    print("\n==== STARTING PRETRAINING ====")
    print(f"Dataset size: {len(pretrain_dataset)}")
    print(f"Block size: {block_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Using device: {next(model.parameters()).device}")

    is_rope = hasattr(model, 'rope') and model.rope
    
    # Use standard learning rate
    actual_lr = pretrain_lr

    # Set up training configuration for pretraining
    tconf = TrainerConfig(
        max_epochs=650,
        batch_size=128,
        learning_rate=actual_lr,
        lr_decay=True,
        warmup_tokens=512*20,
        final_tokens=200*len(pretrain_dataset)*block_size,
        num_workers=0,
        writer=writer
    )

    print(f"PRETRAINING CONFIG: max_epochs={tconf.max_epochs}, batch_size={tconf.batch_size}")
    print(f"Learning rate: {tconf.learning_rate}")
    
    # Create the trainer object
    trainer_obj = Trainer(model, pretrain_dataset, None, tconf)
    
    ### END CODE HERE
    return tconf, trainer_obj

def train(model, writing_params_path, trainer_obj):
    ### TODO:
    ### - Given:
    ###     An output path writing_params_path for the model parameters
    ### [part d]:
    ###
    ### Note: trainer_obj is of type Trainer (see trainer.py for more details)

    ### START CODE HERE
    print("\n==== STARTING TRAINING ====")
    print(f"Trainer configuration: max_epochs={trainer_obj.config.max_epochs}")
    print(f"Dataset size: {len(trainer_obj.train_dataset)}")


    # Train the model using the trainer object
    trainer_obj.train()
    print("\n==== TRAINING COMPLETED ====")
    print(f"Saving model to: {writing_params_path}")
    
    # Save the model parameters to the specified path
    torch.save(model.state_dict(), writing_params_path)
    print(f"Model saved successfully!")


    ### END CODE HERE
    return
