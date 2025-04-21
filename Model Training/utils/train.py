import torch
import os
import logging
import time 
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader, random_split
from utils.data_loader import data_loader
from utils.loss_calculator import calculate_loss

import wandb

def train(
        model,
        device,
        dir_img, 
        dir_label,
        dir_pair, 
        label_suffix, 
        pair_suffix,
        wandb_dir,
        dir_checkpoint,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        val_percent: float = 10,
        save_checkpoint: bool = True,
        
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        wandb_log = False, 
        lr_step_size: int = 10,
        lr_gamma: float = 0.1
):
    
    # 1. Create dataset
    dataset =  data_loader(dir_img, dir_label,dir_pair, label_suffix, pair_suffix)


    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    #loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    # (Initialize logging)
    if wandb_log:
        experiment = wandb.init(project='D-blur_training', resume='allow', anonymous='must', dir = wandb_dir)
        experiment.config.update(
            dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                 val_percent=val_percent, save_checkpoint=save_checkpoint,  amp=amp,
                 dir_img = dir_img, dir_checkpoint = dir_checkpoint, device = device.type)
        )


    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    #criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()  # Start time
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, labels = batch['image'], batch['label']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                
                local_label = labels.to(device=device,  dtype=torch.float32)


                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    # Calculate the MSE based on the non-zero values
                    label_pred = model(images)
                    loss = calculate_loss(label_pred.float(), local_label)


                # loss.backward(retain_graph=True)

                optimizer.zero_grad(set_to_none=False)

                #loss2.backward()
                #logging.info(diffusion_pred.grad)

                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                #logging.info( model.outc_mask.weight.grad)
                
                #logging.info(model.outc_diffusion.weight.grad)

                if wandb_log:
                    experiment.log({
                        'loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if (division_step > 0) & (wandb_log == True):
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())


                        # Log or print the losses
                        logging.info(f'Epoch {epoch}, Loss: {loss:.4f}')
                
                        label = local_label[0].float().cpu()
                        label_loc = torch.sum(label,0)
                        label_dx = torch.sum(label,1)
                        label_prob_pred = torch.sigmoid(label_pred)
                        pred = label_prob_pred[0].float().cpu()
                        pred_loc = torch.sum(pred,0)
                        pred_dx = torch.sum(pred,1)

                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'images channel1': wandb.Image(images[0,0,:,:].cpu()),
                            'images channel2': wandb.Image(images[0,1,:,:].cpu()),
                            'images channel3': wandb.Image(images[0,2,:,:].cpu()),

                            'LocalizationMap': {
                                'true': wandb.Image(label_loc),
                                'pred': wandb.Image(pred_loc),
                            },

                            'DiffusionMapX':{
                                'true': wandb.Image(label_dx),
                                'pred': wandb.Image(pred_dx),
                            },

                            'Validation Loss':loss,
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })
        # Update learning rate after each epoch
        scheduler.step()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        if wandb_log:
            experiment.log({'epoch_duration': epoch_duration, 'epoch': epoch})         
            
        if save_checkpoint:
            # Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # state_dict['mask_values'] = dataset.mask_values
            # Save checkpoint as 'checkpoint_epoch1.pth', 'checkpoint_epoch2.pth', 'checkpoint_epoch3.pth'
            checkpoint_number = (epoch % 3) + 1
            # Convert dir_checkpoint to Path if it's a string
            checkpoint_path = f"{dir_checkpoint}/model_checkpoint_{checkpoint_number}.pth"
            torch.save(state_dict, checkpoint_path)
    
            logging.info(f'Checkpoint for epoch {epoch} saved as checkpoint_epoch{checkpoint_number}.pth!')
            
    return label_pred,labels, images, loss
