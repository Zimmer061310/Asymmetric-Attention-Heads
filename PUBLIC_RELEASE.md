# Public Release Checklist

This private working repository has been cleaned at the tree level, but it
should not be made public directly because old private history may contain
large experiment artifacts, logs, and local metadata. The recommended release
path is a fresh public repository created from the cleaned tree only.

## Recommended Path

1. Wait until the arXiv record is public and stable.
2. Create a fresh public repository, for example `Zimmer061310/AAH-v3`.
3. Export the cleaned tree from this private repository without old Git history.
4. Push that clean tree as the first commit in the public repository.
5. Update `README.md` and `CITATION.cff` with the final arXiv ID and URL.
6. Add the public repository URL to the paper metadata or future revision.

The current private repository can remain the internal development and
experiment-history repository.

## What Must Stay Out Of Git

- `.pt`, `.ckpt`, `.safetensors`, adapter weights, and downloaded model files.
- Raw W&B folders, server logs, tmux logs, and scratch output directories.
- GitHub, W&B, Hugging Face, SSH, and Featurize credentials.
- Server endpoints, passwords, local absolute paths, and personal screenshots.
- Full benchmark caches or datasets that can be regenerated or downloaded.

If checkpoints or adapters are published later, use an artifact store such as
W&B Artifacts, Hugging Face Hub, or an institutional archive, and publish
SHA-256 hashes plus exact model/config revisions.

## Verification Commands

Run these checks on the fresh public release tree before publishing:

```bash
git status --short
find . -name '.DS_Store' -o -name '__pycache__' -o -name '*.pt' -o -name '*.ckpt' -o -name '*.safetensors'
rg -n "ghp_|github_pat_|wandb_|hf_[A-Za-z0-9]|password|pwd:|ssh featurize|workspace\\.featurize|BEGIN OPENSSH"
du -sh .
```

The secret scan should return no real credentials. It is acceptable for
documentation to mention token patterns as examples if no token value appears.

## arXiv Packaging Notes

For arXiv, prefer uploading the TeX source package instead of only the final
PDF. The source package should include:

- the main `.tex` file;
- bibliography as `.bbl` or the `.bib` plus compatible build files;
- all figure files used by the paper;
- any non-standard `.sty` or `.cls` files;
- no private notes, review PDFs, local screenshots, credentials, or raw logs.

As of the current cleanup pass, the final draft PDF was found in the local
Downloads folder, but the matching AAH LaTeX source bundle was not found in
this repository. Do not submit until the matching source bundle has been
exported from Prism or the writing workspace and checked against the final PDF.
