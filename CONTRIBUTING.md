# Contributing to TSInfluenceScoring

Thank you for your interest in contributing to TSInfluenceScoring! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/TSInfluenceScoring.git
cd TSInfluenceScoring
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

3. Run tests to ensure everything is working:
```bash
pytest tests/ -v
```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and modular

## Testing

- Write tests for all new features
- Ensure all tests pass before submitting a PR
- Aim for high test coverage
- Tests should be in the `tests/` directory

Run tests:
```bash
pytest tests/ -v
```

Run tests with coverage:
```bash
pytest tests/ --cov=tsinfluencescoring --cov-report=html
```

## Pull Request Process

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit them:
```bash
git add .
git commit -m "Description of your changes"
```

3. Push to your fork:
```bash
git push origin feature/your-feature-name
```

4. Open a Pull Request with:
   - Clear description of the changes
   - Link to any related issues
   - Test results

## Reporting Issues

When reporting issues, please include:
- Description of the problem
- Steps to reproduce
- Expected behavior
- Actual behavior
- Python version and PyTorch version
- Any relevant error messages or stack traces

## Feature Requests

We welcome feature requests! Please:
- Check if the feature has already been requested
- Provide a clear description of the feature
- Explain the use case and benefits
- Consider contributing the implementation yourself

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help create a positive community

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
