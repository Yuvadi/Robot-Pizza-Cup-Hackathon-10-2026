"""Structured recipe boundary; natural-language interpretation is a future adapter."""

import json

from models import RecipeItem


def parse_recipe(text: str) -> tuple[RecipeItem, ...]:
    """Read {"items": [{"ingredient": "olive", "count": 3}, ...]}."""
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"items"}:
        raise ValueError("Recipe must be an object containing only 'items'.")
    items = value["items"]
    if not isinstance(items, list) or not items:
        raise ValueError("Recipe items must be a nonempty list.")
    result = []
    labels = set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {"ingredient", "count"}:
            raise ValueError("Each recipe item requires exactly ingredient and count.")
        label, count = item["ingredient"], item["count"]
        if not isinstance(label, str) or not label.strip() or label != label.strip():
            raise ValueError("Ingredient labels must be nonempty and have no surrounding whitespace.")
        if type(count) is not int or count < 1:
            raise ValueError("Ingredient counts must be positive integers.")
        if label.casefold() in labels:
            raise ValueError(f"Duplicate ingredient: {label}.")
        labels.add(label.casefold())
        result.append(RecipeItem(label, count))
    return tuple(result)
