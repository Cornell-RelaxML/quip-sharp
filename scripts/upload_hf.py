import argparse

from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument('--public_repo',
                    action='store_false',
                    help='makes repo public')
parser.add_argument('--folder_path', type=str)
parser.add_argument('--repo_id', type=str)
parser.add_argument('--no_multi_commits',
                    action='store_false',
                    help='in chunks for larger uploads')
parser.add_argument('--write_token', type=str)
'''
If fails, just rerun. See https://huggingface.co/docs/huggingface_hub/guides/upload#upload-a-folder-by-chunks
'''

if __name__ == "__main__":
    api = HfApi()
    args = parser.parse_args()
    api.create_repo(
        repo_id=args.repo_id,
        token=args.write_token,
        private=args.public_repo,
        repo_type="model",
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=args.folder_path,
        repo_id=args.repo_id,
        repo_type="model",
        multi_commits=args.no_multi_commits,
        multi_commits_verbose=True,
        token=args.write_token,
        create_pr=True,  # creates a PR. You must manually merge the PR in
    )
