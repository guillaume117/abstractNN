# Contributing to abstractNN

Thank you for your interest in contributing to abstractNN! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Development Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/abstractNN.git
cd abstractNN
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode**

```bash
pip install -e ".[dev,docs]"
```

4. **Install pre-commit hooks**

```bash
pip install pre-commit
pre-commit install
```

## 🔧 Development Workflow

### Creating a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number-description
```

### Making Changes

1. **Write code** following the style guide
2. **Add tests** for new functionality
3. **Update documentation** if needed
4. **Run tests** to ensure everything works

### Code Style

We use the following tools for code quality:

```bash
# Format code
black abstractnn/ tests/
isort abstractnn/ tests/

# Lint code
flake8 abstractnn/ tests/

# Type checking
mypy abstractnn/
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=abstractnn --cov-report=html

# Run specific test file
pytest tests/test_affine_engine.py -v

# Run with markers
pytest tests/ -m "not slow"
```

### Building Documentation

```bash
cd docs
make html
# View at docs/_build/html/index.html
```

## 📝 Commit Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

