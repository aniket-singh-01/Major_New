# Git Large File Cleanup Guide

You need to remove the large files (especially `optimized_model.keras` at 114.30 MB) from your Git history to be able to push to GitHub.

## Method 1: Automated Script

You can use the included bash script:

```bash
# On Windows with Git Bash
bash git_cleanup_script.sh

# On Linux/Mac
chmod +x git_cleanup_script.sh
./git_cleanup_script.sh
```

## Method 2: Manual Steps

If you prefer to run the commands yourself:

1. **Remove from current tracking**:
   ```
   git rm --cached optimized_model.keras
   git rm --cached *.h5
   git rm --cached *.keras
   ```

2. **Commit the removal**:
   ```
   git add .gitignore
   git commit -m "Updated .gitignore and removed large files from tracking"
   ```

3. **Purge from Git history**:
   ```
   git filter-branch --force --index-filter "git rm --cached --ignore-unmatch optimized_model.keras" --prune-empty --tag-name-filter cat -- --all
   git filter-branch --force --index-filter "git rm --cached --ignore-unmatch *.h5" --prune-empty --tag-name-filter cat -- --all
   git filter-branch --force --index-filter "git rm --cached --ignore-unmatch *.keras" --prune-empty --tag-name-filter cat -- --all
   ```

4. **Clean up references**:
   ```
   git for-each-ref --format="delete %(refname)" refs/original | git update-ref --stdin
   git reflog expire --expire=now --all
   git gc --prune=now --aggressive
   ```

5. **Force push to GitHub**:
   ```
   git push --force origin Major
   ```

## Method 3: Git LFS (For Future Use)

For future large files, consider using Git LFS:

1. **Install Git LFS**:
   ```
   git lfs install
   ```

2. **Track large file types**:
   ```
   git lfs track "*.h5"
   git lfs track "*.keras"
   git lfs track "*.tflite"
   ```

3. **Add tracking configuration**:
   ```
   git add .gitattributes
   git commit -m "Add Git LFS tracking for large model files"
   ```

4. **Add and commit your large files normally**:
   ```
   git add your_large_file.h5
   git commit -m "Add large model file using Git LFS"
   ```

## Sharing Model Files

Instead of tracking large model files in Git, consider:

1. Uploading models to a cloud storage service (Google Drive, Dropbox)
2. Using a model hub like TensorFlow Hub or Hugging Face
3. Including scripts to download the models from external sources
4. Creating smaller TFLite versions for sharing directly
