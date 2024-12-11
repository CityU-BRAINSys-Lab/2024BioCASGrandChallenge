import os
import sys
import json
import csv
import torch
from datetime import datetime


class TrainingCheckpointLogger():
    def __init__(self, args):
        training_starttime = datetime.now().strftime("%d_%m_%Y-%H_%M_%S-%f")
        assert (len(args.all_files) == 1) == ("recwise" in args.result_dir)
        
        if not os.path.exists(args.result_dir):
            response = input(f"The result directory {args.result_dir} does not exist. Do you want to create it? (y/n): ")
            if response.lower() != "y":
                sys.exit(0)
        
        if (len(args.all_files) == 1) & ("recwise" in args.result_dir):
            self.ckpt_dir = os.path.join(args.result_dir, args.network, args.all_files[0],
                            f'checkpoints_{training_starttime}')
        else:
            self.ckpt_dir = os.path.join(args.result_dir, args.network,
                            f'checkpoints_{training_starttime}')
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.log_keys = ['epoch', 'train_loss', 'valid_loss', 'valid_r2']
        self.csvlogger = CSVLogger(self.log_keys, os.path.join(self.ckpt_dir, 'log.csv'), append=True)

        self.info_file = os.path.join(self.ckpt_dir, "proj_info.json")
        with open(self.info_file, 'w') as f:
            json.dump(vars(args), f, indent=4)
    
    def write_log(self, logs):
        self.csvlogger.write(logs)

    def save_model(self, model, ckpt_name):
        torch.save(model.state_dict(),
            # {
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            # },
            os.path.join(self.ckpt_dir, ckpt_name),
        )
        
    def add_info(self, **info):
        """
        Write the project information to a json file.
        info: dict including the additional information to be written.
        """
        with open(self.info_file, 'r') as f:
            info = json.load(f) | info
        with open(self.info_file, 'w') as f:
            json.dump(info, f, indent=4)


class CSVLogger(object):
    def __init__(self, keys, path, append=False):
        super(CSVLogger, self).__init__()
        self._keys = keys
        self._path = path
        if append is False or not os.path.exists(self._path):
            with open(self._path, 'w') as f:
                w = csv.DictWriter(f, self._keys)
                w.writeheader()

    def write(self, logs):
        with open(self._path, 'a') as f:
            w = csv.DictWriter(f, self._keys)
            w.writerow(logs)

    def get_ckpt_dir(self):
        return os.path.dirname(self._path)
    
    def get_column(self, key):
        with open(self._path, 'r') as f:
            reader = csv.DictReader(f)
            return [row[key] for row in reader]