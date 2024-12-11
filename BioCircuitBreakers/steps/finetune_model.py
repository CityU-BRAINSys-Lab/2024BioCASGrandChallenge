from tqdm import tqdm

import torch

def FinetuneModel(args, data, model, preprocessors:list, augmentators:list, device, training_logger):
    # Set training hyperparameters
    total_epoch = args.finetune_num_epochs
    lr = 1e-4

    # Get the data loader
    train_set_loader, valid_set_loader = data

    # Set the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    # move the model to device
    model.to(device)

    # Record the training process
    sum_train_losses = 0.0

    # loop over the epochs
    for epoch in range(total_epoch):
        # train
        sum_train_losses = 0.0
        model.train()
        for dl in [train_set_loader, valid_set_loader]:
            for i, sample in tqdm(enumerate(dl), 
                            desc=f"Epoch {epoch}/{total_epoch - 1}",
                            total=len(dl)):
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
                optimizer.step()

                sum_train_losses += loss.item() * x.size(0)

        train_loss = sum_train_losses / (len(train_set_loader.dataset) + len(valid_set_loader.dataset))

        # Print the training process
        print(f"Epoch {epoch}/{total_epoch - 1}: ",
                f"LR: {lr}, train_loss: {train_loss:.4f}")

        # Overwrite the latest model
        training_logger.save_model(model, "ft_latest.pth")
    
    print("Finetuning finished.")