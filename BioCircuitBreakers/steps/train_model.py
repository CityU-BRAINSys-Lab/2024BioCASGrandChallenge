from tqdm import tqdm

import torch
import numpy as np

from neurobench.benchmarks.workload_metrics import r2


def TrainModel(args, data, model, preprocessors:list, augmentators:list, device, training_logger):
    # Set training hyperparameters
    total_epoch = args.train_num_epochs
    lr = 1e-3

    # Get the data loader
    train_set_loader, valid_set_loader = data

    # Get output dimension
    output_dim = args.model_output_dim

    # Set the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch)

    # Instantiate the metric
    metric = r2()

    # Record the best valid r2 score
    best_valid_r2 = -np.inf

    # move the model to device
    model.to(device)

    # Record the training process
    sum_train_losses = 0.0
    sum_valid_losses = 0.0

    # loop over the epochs
    for epoch in range(total_epoch):
        # train
        sum_train_losses = 0.0
        model.train()
        for i, sample in tqdm(enumerate(train_set_loader), 
                           desc=f"Epoch {epoch}/{total_epoch - 1}",
                           total=len(train_set_loader)):
            for a in augmentators:
                sample = a(sample)
            for p in preprocessors:
                sample = p(sample)
            x = torch.flip(sample[0], dims=[1]).to(device)
            y = torch.flip(sample[1], dims=[1]).to(device) if args.dataset_label_series else sample[1]
            
            o = model(x)

            loss = model.loss_function(o, (x,y))
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            sum_train_losses += loss.item()

        # if scheduler.get_last_lr()[0] > 5e-4:
        scheduler.step()
        train_loss = sum_train_losses / len(train_set_loader)

        # validate no validation loss data if using benchmark
        with torch.no_grad():

            sum_valid_losses = 0.0
            metric.reset()
            model.eval()
            for sample in valid_set_loader:
                for p in preprocessors:
                    sample = p(sample)
                x = torch.flip(sample[0], dims=[1]).to(device)
                y = torch.flip(sample[1], dims=[1]).to(device) if args.dataset_label_series else sample[1]

                o = model(x)

                loss = model.loss_function(o, (x,y)) * x.size(0)
                sum_valid_losses += loss.item()
                
                y_pred = model.post_process(o)
                valid_r2 = metric(model, y_pred.view(-1, output_dim), (x,y.view(-1, output_dim)))

            val_loss = sum_valid_losses / len(valid_set_loader.dataset)

        # Print the training process
        print(f"Epoch {epoch}/{total_epoch - 1}: ",
                f"LR: {scheduler.get_last_lr()}, train_loss: {train_loss:.4f}, \
                valid_loss: {val_loss:.4f}, valid_r2: {valid_r2:.4f}")

        # Write to the csv log
        logs = dict()
        logs['epoch'] = epoch
        logs['train_loss'] = train_loss
        logs['valid_loss'] = val_loss
        logs['valid_r2'] = valid_r2
        training_logger.write_log(logs)

        # Save the model
        if valid_r2 > best_valid_r2:
            best_valid_r2 = valid_r2
            best_epoch = epoch
            training_logger.save_model(model, f"ckpt_epoch_{epoch}.pth")
            
        # Overwrite the latest model
        training_logger.save_model(model, "latest.pth")
    
    args.add_attr(best_epoch=best_epoch, best_valid_r2=best_valid_r2)
    training_logger.add_info(best_epoch=best_epoch, best_valid_r2=best_valid_r2)
    print("Training finished.")
    print(f"Best valid r2: {best_valid_r2} at epoch {best_epoch}.")