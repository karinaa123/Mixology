import anthropic

client = anthropic.Anthropic(api_key="")

# Try different models
models_to_test = [
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-5-20250929",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-opus-4-20250514"
]

for model in models_to_test:
    try:
        message = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        print(f"✓ {model} - WORKS")
    except Exception as e:
        print(f"✗ {model} - {e}")