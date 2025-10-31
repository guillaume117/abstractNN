# Publishing Guide for abstractNN

This guide explains how to publish abstractNN to PyPI.

## Prerequisites

1. **PyPI Account**
   - Create account at https://pypi.org
   - Create account at https://test.pypi.org (for testing)

2. **API Tokens**
   - Generate API token at https://pypi.org/manage/account/token/
   - Generate API token at https://test.pypi.org/manage/account/token/

3. **Tools**
   ```bash
   pip install build twine
   ```

## Publishing Process

### 1. Prepare Release

```bash
# Run preparation script
./scripts/prepare_release.sh 0.1.1

# Review changes
git diff

# Commit version bump
git add .
git commit -m "chore: bump version to 0.1.1"

# Create tag
git tag -a v0.1.1 -m "Release v0.1.1"

# Push
git push origin main
git push origin v0.1.1
```

### 2. Test on Test PyPI

```bash
# Configure Test PyPI credentials
# Create ~/.pypirc:
cat > ~/.pypirc << EOF
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_PRODUCTION_TOKEN

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TEST_TOKEN
EOF

chmod 600 ~/.pypirc

# Build and upload to Test PyPI
python -m build
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ abstractNN==0.1.1

# Test that it works
python -c "from abstractnn import AffineEngine; print('✅ Import successful')"
```

### 3. Publish to PyPI

```bash
# Upload to production PyPI
twine upload dist/*

# Verify on PyPI
# Visit: https://pypi.org/project/abstractNN/

# Test installation
pip install abstractNN==0.1.1

# Verify
python -c "from abstractnn import AffineEngine; print('✅ Package published successfully')"
```

### 4. Create GitHub Release

1. Go to https://github.com/flyworthi/abstractNN/releases
2. Click "Draft a new release"
3. Choose tag `v0.1.1`
4. Title: `v0.1.1 - Description`
5. Copy changelog entry
6. Attach built wheels from `dist/`
7. Publish release

## Automated Publishing (GitHub Actions)

The repository includes automated publishing via GitHub Actions.

### Setup Secrets

In GitHub repository settings, add:
- `PYPI_API_TOKEN`: Production PyPI token
- `TEST_PYPI_API_TOKEN`: Test PyPI token

### Trigger Automatic Publish

```bash
# Create and push tag
git tag v0.1.1
git push origin v0.1.1

# Create GitHub release
# GitHub Actions will automatically publish to PyPI
```

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **Major** (1.0.0): Breaking changes
- **Minor** (0.1.0): New features, backwards compatible
- **Patch** (0.0.1): Bug fixes

## Pre-release Checklist

- [ ] All tests pass
- [ ] Code formatted (black, isort)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in all files
- [ ] Git tag created
- [ ] Tested on Test PyPI

## Post-release Checklist

- [ ] Package appears on PyPI
- [ ] Installation works (`pip install abstractNN`)
- [ ] GitHub release created
- [ ] Documentation deployed
- [ ] Announcement posted (Twitter, blog, etc.)

## Troubleshooting

### Upload Fails

```bash
# Check package
twine check dist/*

# Verify version doesn't exist
pip index versions abstractNN
```

### Import Errors After Install

```bash
# Check installed files
pip show -f abstractNN

# Reinstall
pip uninstall abstractNN
pip install abstractNN --no-cache-dir
```

### Documentation Not Updated

```bash
# Trigger ReadTheDocs build
# Go to: https://readthedocs.org/projects/abstractnn/builds/
```

## Rolling Back

If a release has issues:

```bash
# Yank the bad release (doesn't delete, just hides)
# On PyPI website, go to release and click "Yank"

# Or via API
pip install yank
yank -v 0.1.1 "Reason for yanking"
```

## Contact

For publishing issues, contact:
- Email: contact@flyworthi.ai
- GitHub: @flyworthi
