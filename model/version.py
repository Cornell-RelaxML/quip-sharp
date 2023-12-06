MODEL_VERSION = 0

def check_model_version(test):
    if test != MODEL_VERSION:
        raise Exception(
            f"Saved model version ({test}) does not match the "\
            f"source code model version ({MODEL_VERSION}). "\
            "Please pull the latest code from git@github.com:Cornell-RelaxML/quip-sharp.git")
