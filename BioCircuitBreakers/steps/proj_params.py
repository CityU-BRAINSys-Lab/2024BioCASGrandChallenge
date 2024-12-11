import os
import json
import importlib
from argparse import ArgumentParser


class ProjectParams():
    def __init__(self, json_file_dir=None, **kwargs):
        if json_file_dir:
            with open(os.path.join(json_file_dir, "proj_info.json"), "r") as f:
                json_info = json.load(f)
            self._load_arguments(**json_info)
            self.update_attr(**kwargs)
            self.update_attr(result_dir=json_file_dir)
        else:
            self._load_arguments(**kwargs)
            self._load_hyperparams()

    def _load_arguments(self, **kwargs):
        if not kwargs:
            parser = ArgumentParser()
            parser.add_argument("-d", "--data_dir", type=str, 
                                default="neurobench/data/primate_reaching/PrimateReachingDataset/",
                                help="The directory of the data.")
            parser.add_argument("-m", "--monkey", type=str,
                                choices=["indy", "loco"],
                                required=True,
                                help="The name of monkey to train on.")
            parser.add_argument("-r", "--result_dir", type=str,
                                required=True,
                                help="The directory to save the results.")
            parser.add_argument("-n", "--network", type=str,
                                required=True,
                                help="The network to train.")
            parser.add_argument("-a", "--all_files", type=str, nargs="+", 
                                default=["indy_20160622_01", "indy_20160630_01", "indy_20170131_02", 
                                        "loco_20170210_03", "loco_20170215_02", "loco_20170301_05"],
                                help="The files to load.")
            args = parser.parse_args()  

            for k, v in vars(args).items():
                setattr(self, k, v)

        else:
            for k, v in kwargs.items():
                setattr(self, k, v)


    def _load_hyperparams(self):
        # Default hyperparams
        self.preprocessors = []
        self.augmentators = []

        hyperparams = importlib.import_module(f"hyperparams.{self.network}")
        for k, v in vars(hyperparams).items():
            if not k.startswith("__"):
                setattr(self, k.lower(), v)

    def print(self):
        print("=====Project Parameters=====")
        for k, v in vars(self).items():
            print(f"{k}: {v}")
        print("=============================")

    def add_attr(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)
            else:
                raise AttributeError(f"Attribute {k} already exists in the project parameters.")

    def update_attr(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise AttributeError(f"Attribute {k} does not exist in the project parameters.")
    
    def update_json_file(self):
        from deepdiff import DeepDiff
        json_file_path = os.path.join(self.result_dir, "proj_info.json")
        if os.path.exists(json_file_path):
            with open(os.path.join(self.result_dir, "proj_info.json"), "r") as f:
                ori_json_info = json.load(f)
            curr_json_info = vars(self)
            diff = DeepDiff(ori_json_info, curr_json_info)
            if 'dictionary_item_removed' in diff or 'values_changed' in diff:
                print("--------------Original json file:------------")
                print(ori_json_info)
                print("--------------Current json file:------------")
                print(curr_json_info)
                print("--------------Difference:------------") 
                print(diff)
                response = input("Json file will probably discard some information as above. Do you really want to overwrite?  (y/n): ")
                if response.lower() != "y":
                    return
            with open(os.path.join(self.result_dir, "proj_info.json"), "w") as f:
                json.dump(curr_json_info, f, indent=4)
                print(f"Json file '{json_file_path}' updated.")
        else:
            raise FileNotFoundError(f"File '{json_file_path}' does not exist.",
                                    "This method only for existing json file update.",
                                    "Please create new json file through only during training.")

    def get_model_arch_params(self):
        return {'_'.join(k.split('_')[1:]): v for k, v in vars(self).items() if k.startswith('model_')}
    