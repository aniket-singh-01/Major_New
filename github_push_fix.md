# GitHub Push Error Fix Guide

You're encountering an SSL error when pushing to GitHub, which is often caused by:
1. Network connectivity issues
2. Pushing large files (GitHub has a 100MB file size limit)
3. Git or SSL configuration problems

## Step 1: Remove Large Files from Git Tracking

First, let's make sure all large files are properly excluded:

```bash
# Check if any large files are still being tracked
git ls-files --stage | grep -v "^120000" | awk '{print $4, $1, $2, $3}' | sort -k 1 -n -r | head -20
```

If you see `optimized_model.keras` or any other large files still being tracked, remove them:

```bash
git rm --cached optimized_model.keras
git rm --cached *.h5
git commit -m "Remove large files from tracking"
```

## Step 2: Fix Network-Related Issues

The SSL error suggests network issues. Try these solutions:

### Increase Git Buffer Size

```bash
git config --global http.postBuffer 524288000
```

### Disable SSL Verification (only if you're on a trusted network)

```bash
git config --global http.sslVerify false
```

### Use SSH Instead of HTTPS

If you're using HTTPS, try switching to SSH:

1. Generate an SSH key (if you don't have one):
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```

2. Add the key to your GitHub account (copy the output of):
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

3. Change your remote URL:
   ```bash
   git remote set-url origin git@github.com:aniket-singh-01/Major_New.git
   ```

## Step 3: Push with Smaller Chunks

Try pushing with the `--no-thin` option which sends complete objects:

```bash
git push --no-thin origin Major
```

Or push with compression disabled:

```bash
git config --global core.compression 0
git push origin Major
```

## Step 4: Last Resort - Local LFS

If you still need to track large files, install Git LFS locally:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.keras"
git lfs track "optimized_model.*"

# Add .gitattributes
git add .gitattributes
git commit -m "Setup Git LFS tracking"

# Try pushing again
git push origin Major
```

## Still Having Issues?

If you continue to face problems, consider:

1. Uploading your code without the model files
2. Sharing model files separately via Google Drive or other file sharing services
3. Creating a small dummy model for demonstration purposes
