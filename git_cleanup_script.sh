#!/bin/bash

# This script helps clean up large files from Git history

echo "Git Large File Cleanup Script"
echo "==============================="
echo "This script will remove large files from your Git history."
echo "Please make sure you have a backup of your repository before proceeding."
echo ""

# Remove the large file from current tracking
echo "Step 1: Removing large files from current tracking..."
git rm --cached optimized_model.keras
git rm --cached *.h5
git rm --cached *.keras

# Commit these changes
echo "Step 2: Committing changes to .gitignore and file removals..."
git add .gitignore
git commit -m "Updated .gitignore and removed large files from tracking"

# Purge the files from Git history
echo "Step 3: Removing large files from Git history..."
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch optimized_model.keras" --prune-empty --tag-name-filter cat -- --all
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch *.h5" --prune-empty --tag-name-filter cat -- --all
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch *.keras" --prune-empty --tag-name-filter cat -- --all

# Clean up and garbage collect
echo "Step 4: Cleaning up references and garbage collecting..."
git for-each-ref --format="delete %(refname)" refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "Large files have been removed from Git history."
echo "You can now push with: git push --force origin Major"
echo ""
echo "NOTE: Be careful with --force, as it will overwrite the remote history."
echo "Make sure other collaborators are aware of this change."
