import os
import yaml
import shutil

class MyDumper(yaml.SafeDumper):
    """Custom YAML dumper to enforce block formatting."""
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow=flow, indentless=False)

config_dir = "/home/ss15859/Documents/EarthquakeNPP/Experiments/AutoSTPP/configs"

for filename in os.listdir(config_dir):
    if filename.endswith(".yaml") and "ComCat" in filename:
        file_path = os.path.join(config_dir, filename)
        
        # Backup original file
        # shutil.copy(file_path, file_path + ".bak")

        try:
            # Load the YAML file
            with open(file_path, "r") as file:
                config = yaml.safe_load(file) or {}

            # create a new key model:init_args:background_rate
            config["catalog"]["Mcut"] = 2.4
            config["catalog"]["auxiliary_start"] = "1971-01-01 00:00:00"
            config["catalog"]["train_nll_start"] = "1986-01-01 00:00:00"
            config["catalog"]["val_nll_start"] = "1999-01-01 00:00:00"
            config["catalog"]["test_nll_start"] = "2013-01-01 00:00:00"
            config["catalog"]["test_nll_end"] = "2024-01-01 00:00:00"
            config["catalog"]["path"] = "../../Datasets/China/CSEP_CN_catalog.csv"
            config["catalog"]["path_to_polygon"] = "CSEP_CN_shape.npy"


            # add "_background" beween "stpp" and "_seed" in the filename
            file_path = file_path.replace("ComCat_25", "China_24")
            filename = filename.replace("ComCat_25", "China_24")

            # Ensure necessary keys exist and update 'experiment' field
            config.setdefault("trainer", {}).setdefault("logger", {}).setdefault("init_args", {})["experiment"] = os.path.splitext(filename)[0]
            config.setdefault("model", {}).setdefault("init_args", {})["name"] = os.path.splitext(filename)[0]
            config.setdefault("data", {}).setdefault("init_args", {})["name"] = os.path.splitext(filename)[0]

            # Force "devices: [0]"
            # config["trainer"]["devices"] = [0]

            # Save the updated YAML file
            with open(file_path, "w") as file:
                yaml.dump(config, file, Dumper=MyDumper, default_flow_style=False, sort_keys=False, indent=4)

            print(f"Updated {filename} successfully.")
        except yaml.YAMLError as e:
            print(f"Error processing {filename}: {e}")
        except Exception as e:
            print(f"Unexpected error with {filename}: {e}")

print("All config files processed.")
