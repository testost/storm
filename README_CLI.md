# STORM CLI Documentation

## Installation
```bash
# Install dependencies
poetry install

# Verify installation
poetry run python -c "import dspy"
```

## Usage
```bash
# Launch the CLI
storm

# Direct command execution
storm <<< "/help"  # Show help
```

### Key Commands
| Command       | Description                          |
|---------------|--------------------------------------|
| `/help`       | Show all available commands          |
| `/exit`       | Exit the CLI                         |
| `/open`       | Open last generated article          |
| `/list`       | Browse and open previous articles    |
| `/clear`      | Clear terminal screen                |
| `/version`    | Show CLI version                     |

## Bash Shortcuts
Add to your `~/.bashrc`:
```bash
# Quick access commands
alias stormlist='storm <<< "/list"'
alias stormopen='storm <<< "/open"'
```

## Demo Limitations
- Based on GPT-4 API (requires valid API key)
- Rate limited to 3 requests/minute
- Article length capped at 2000 tokens
- No persistent history between sessions
- Articles stored locally in `results/gpt/`

## Configuration
Required environment variables:
```bash
# API Keys
export OPENAI_API_KEY='your-key-here'
export YDC_API_KEY='your-key-here'  # For demo search
```

## Troubleshooting
```bash
# Reset environment if issues occur
poetry env remove python3.10
rm poetry.lock
poetry install

# Check installation
poetry env info
poetry show | grep "dspy\|json-repair"
```

### Known Issues
1. **Pydantic Warning**
   ```
   pydantic/_internal/_config.py: UserWarning: Valid config keys have changed in V2:
   * 'fields' has been removed
   ```
   This is a known warning from pydantic V2 and doesn't affect functionality. To suppress it, you can:
   ```python
   import warnings
   warnings.filterwarnings("ignore", message="Valid config keys have changed in V2")
   ```

## File Structure
```
results/gpt/
└── topic_name_timestamp/
    ├── storm_gen_article.md   # Markdown version
    └── storm_gen_article.txt  # Plain text version
