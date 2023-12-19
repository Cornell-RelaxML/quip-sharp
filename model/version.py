MODEL_VERSION = 1

def check_model_version(test):
    if test != MODEL_VERSION:
        raise Exception(
            f"Saved model version ({test}) does not match the "\
            f"source code model version ({MODEL_VERSION}). "\
            "Please pull the latest code or model checkpoints.")
