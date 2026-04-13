# SCIV Documentation

This directory contains the documentation for SCIV (Single-Cell Integrated Variational) tool.

## Documentation Structure

- `index.md` - Documentation homepage
- `sciv_usage.md` - Usage guide and API reference
- `_config.yml` - Jekyll configuration file for static site generation

## Building the Documentation

To build the documentation locally, you'll need Jekyll installed. Follow these steps:

1. Install Ruby and Jekyll:
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install ruby-full build-essential zlib1g-dev
   
   # On macOS
   brew install ruby
   
   # Install Jekyll and bundler
   gem install jekyll bundler
   ```

2. Navigate to the docs directory:
   ```bash
   cd docs
   ```

3. Install dependencies:
   ```bash
   bundle install
   ```

4. Build and serve the documentation:
   ```bash
   bundle exec jekyll serve
   ```

5. Open your browser and go to `http://localhost:4000` to view the documentation.

## Contributing to Documentation

To contribute to the documentation, simply edit the Markdown files in this directory. The documentation is built using Jekyll and the Just the Docs theme.

### Documentation Guidelines

- Use clear, concise language
- Include code examples where appropriate
- Follow the existing formatting and structure
- Add new sections as needed for new features

## License

The documentation is released under the same license as SCIV itself.