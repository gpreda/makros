#!/usr/bin/env python3
"""Makros - A macros diet tracker CLI."""

import ast
import json
import os
import sys

from google import genai

from models import VALID_UNITS


def get_client() -> genai.Client:
    """Get Gemini client with API key from config."""
    config_file = os.path.expanduser('~/.config/tongue/config.json')

    # Check environment variable first
    api_key = os.environ.get('GEMINI_API_KEY')

    # Fall back to config file
    if not api_key:
        if not os.path.exists(config_file):
            print(f"Error: Config file not found at {config_file}")
            print('Please create it with: {"gemini_api_key": "YOUR_API_KEY_HERE"}')
            sys.exit(1)

        with open(config_file, 'r') as f:
            config = json.load(f)

        api_key = config.get('gemini_api_key')
        if not api_key:
            print("Error: gemini_api_key not found in config file")
            sys.exit(1)

    return genai.Client(api_key=api_key)


def analyze_meal(meal_description: str) -> dict:
    """Send meal description to Gemini and get nutritional breakdown."""
    client = get_client()

    units_list = ', '.join(VALID_UNITS)
    prompt = f"""
Analyze this meal and provide a detailed nutritional breakdown:

Meal: "{meal_description}"

For each ingredient/item in the meal, provide:
- name: ingredient name (string)
- amount: numeric quantity (float)
- unit: one of [{units_list}]
- calories: kcal (int)
- protein: grams (float)
- carbs: grams (float)
- fat: grams (float)
- fiber: grams (float)
- alcohol: grams (float) - only for alcoholic beverages, 0 otherwise

Respond with ONLY a Python dictionary in this exact format:
{{
    'items': [
        {{'name': 'ingredient name', 'amount': 100.0, 'unit': 'g', 'calories': 123, 'protein': 12.5, 'carbs': 10.0, 'fat': 8.0, 'fiber': 1.0, 'alcohol': 0.0}},
        ...
    ],
    'totals': {{'calories': 456, 'protein': 25.0, 'carbs': 30.0, 'fat': 20.0, 'fiber': 3.0, 'alcohol': 0.0}}
}}

Be accurate with portion sizes. Use standard nutritional databases as reference.
Return ONLY the dictionary, no other text or markdown.
"""

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt
    )
    text = response.text.strip()

    # Clean up response
    text = text.replace('```python', '').replace('```', '').strip()

    try:
        result = ast.literal_eval(text)
        return result
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing response: {e}")
        print(f"Raw response:\n{text}")
        sys.exit(1)


def print_table(data: dict) -> None:
    """Print nutritional data as a formatted table."""
    items = data.get('items', [])
    totals = data.get('totals', {})

    # Column headers and widths
    headers = ['Item', 'Amount', 'Unit', 'Kcal', 'Protein', 'Carbs', 'Fat', 'Fiber', 'Alc']
    widths = [20, 8, 8, 6, 8, 8, 8, 8, 6]

    # Print header
    header_line = ''
    for h, w in zip(headers, widths):
        header_line += h.ljust(w)
    print(header_line)
    print('-' * sum(widths))

    # Print items
    for item in items:
        amount = item.get('amount', 0)
        amount_str = f"{amount:.1f}" if isinstance(amount, float) else str(amount)
        row = [
            item.get('name', '')[:19],
            amount_str,
            item.get('unit', ''),
            str(item.get('calories', 0)),
            f"{item.get('protein', 0):.1f}",
            f"{item.get('carbs', 0):.1f}",
            f"{item.get('fat', 0):.1f}",
            f"{item.get('fiber', 0):.1f}",
            f"{item.get('alcohol', 0):.1f}"
        ]
        row_line = ''
        for val, w in zip(row, widths):
            row_line += str(val).ljust(w)
        print(row_line)

    # Print totals
    print('-' * sum(widths))
    totals_row = [
        'TOTAL',
        '',
        '',
        str(totals.get('calories', 0)),
        f"{totals.get('protein', 0):.1f}",
        f"{totals.get('carbs', 0):.1f}",
        f"{totals.get('fat', 0):.1f}",
        f"{totals.get('fiber', 0):.1f}",
        f"{totals.get('alcohol', 0):.1f}"
    ]
    totals_line = ''
    for val, w in zip(totals_row, widths):
        totals_line += str(val).ljust(w)
    print(totals_line)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python makros.py <meal description>")
        print("Example: python makros.py '4 eggs, 50g cheese, 2 slices of bread'")
        sys.exit(1)

    meal = ' '.join(sys.argv[1:])
    print(f"\nAnalyzing: {meal}\n")

    data = analyze_meal(meal)
    print_table(data)


if __name__ == '__main__':
    main()
