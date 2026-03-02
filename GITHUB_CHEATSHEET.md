# GitHub Upload Cheat Sheet — Lead Scoring Project

## Pre-Push Security Check (Run These First)

```bash
# 1. Verify no secrets in git history
git log --all --full-history -- .env */*.json | head -20

# 2. Check no large data files tracked
git ls-files | xargs -I{} sh -c 'du -h "{}" 2>/dev/null' | sort -hr | head -10

# 3. Scan for potential hardcoded secrets
grep -r "sk-.*\|api_key.*=\|password.*=\|AKIA.*" --include="*.py" . 2>/dev/null
```

## Initial Upload to GitHub

```bash
# 1. Initialize git (if not already)
git init

# 2. Add all files respecting .gitignore
git add .

# 3. Check what's being added (REVIEW THIS!)
git status
git diff --cached --stat

# 4. Commit
git commit -m "Initial commit: Lead scoring system with FastAPI, Streamlit, GCP deployment

Features:
- ML model (Logistic Regression, AUC ~0.83)
- SHAP explainability for per-lead insights
- FastAPI with batch CSV scoring
- Streamlit dashboard for portfolio showcase
- GCP Cloud Run deployment config
- Docker & docker-compose for local dev"

# 5. Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/lead-scoring.git

# 6. Push to main branch
git push -u origin main
# OR if your default branch is master:
git push -u origin master
```

## If You Have an Existing Repo

```bash
# Force push (ONLY for new repos, not shared ones!)
git push -u origin main --force-with-lease

# Or resolve conflicts if repo has existing content
git pull origin main --rebase
git push origin main
```

## Post-Push Verification

```bash
# Check GitHub has correct files
curl -s https://api.github.com/repos/YOUR_USERNAME/lead-scoring/contents/ | grep name

# Or just browse to:
# https://github.com/YOUR_USERNAME/lead-scoring
```

## Common Issues & Fixes

**Issue: `git push` asks for password repeatedly**
```bash
# Use HTTPS with token or switch to SSH
git remote set-url origin git@github.com:YOUR_USERNAME/lead-scoring.git
# Then: https://github.com/settings/tokens -> Generate new token (classic)
# Select scopes: repo, read:org
```

**Issue: Large files rejected (model.pkl > 100MB)**
```bash
# Untrack and ignore
git rm --cached models/lead_scorer_v1.pkl
echo "models/*.pkl" >> .gitignore
git commit -m "Remove large model file from tracking"
git push

# Or use Git LFS for large files
git lfs track "*.pkl"
git add .gitattributes
git push
```

**Issue: Accidentally committed secrets**
```bash
# DANGER: Rewrites history - coordinate if team project
git filter-repo --path .env --invert-paths
# OR use BFG Repo-Cleaner
# Then force push
```

## Quick Reference

| Task | Command |
|------|---------|
| Check status | `git status` |
| See what will be pushed | `git diff --stat origin/main..HEAD` |
| Undo last commit (keep changes) | `git reset --soft HEAD~1` |
| Undo add of specific file | `git reset HEAD path/to/file` |
| View commit history | `git log --oneline -10` |
| Push specific branch | `git push origin feature-branch` |

## Repository Settings on GitHub (After Push)

1. **Add README header image** (optional):
   - Go to repo → Settings → Social preview → Upload image

2. **Topics** (helps discoverability):
   - Click gear icon next to "About" → Add topics:
   - `machine-learning`, `lead-scoring`, `fastapi`, `streamlit`, `gcp`, `shap`, `mlops`

3. **Make it a template** (optional):
   - Settings → General → Template repository

4. **Enable GitHub Pages** (for docs):
   - Settings → Pages → Source: Deploy from a branch → main /docs

## One-Liner Test After Push

```bash
# Clone to temp location and verify it works
cd /tmp && git clone https://github.com/YOUR_USERNAME/lead-scoring.git && cd lead-scoring && ls -la
```
