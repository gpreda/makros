"""Data models for makros."""

from dataclasses import dataclass, field
from typing import Optional


# Units that are universally convertible (weight-based)
WEIGHT_UNITS = {'g', 'kg', 'oz', 'lb'}

# Units that are universally convertible (volume-based)
VOLUME_UNITS = {'ml', 'l', 'fl_oz'}

# Units that require item-specific conversion factors to grams
ITEM_DEPENDENT_UNITS = {'tbsp', 'tsp', 'cup', 'item', 'slice', 'serving'}

# All valid units
VALID_UNITS = WEIGHT_UNITS | VOLUME_UNITS | ITEM_DEPENDENT_UNITS

# Universal conversion factors to base unit (grams for weight, ml for volume)
WEIGHT_TO_GRAMS = {
    'g': 1.0,
    'kg': 1000.0,
    'oz': 28.3495,
    'lb': 453.592,
}

VOLUME_TO_ML = {
    'ml': 1.0,
    'l': 1000.0,
    'fl_oz': 29.5735,
}


@dataclass
class Item:
    """A food item with nutritional info and unit conversions.

    Attributes:
        id: Auto-increment unique identifier (None for new items)
        bar_code: Unique barcode (can be None)
        name: Unique item name
        description: Optional description
        unit_conversions: Dict mapping item-dependent units to grams
                         e.g., {'tbsp': 13.5, 'item': 50.0}
        default_unit: The unit for nutritional info (e.g., 'item', 'g', 'serving')
        calories/protein/carbs/fat/fiber/alcohol: Nutrition per default_unit
    """
    name: str
    id: Optional[int] = None
    bar_code: Optional[str] = None
    description: Optional[str] = None
    unit_conversions: dict[str, float] = field(default_factory=dict)
    # Nutritional info per default_unit
    default_unit: str = 'item'
    calories: Optional[float] = None
    protein: Optional[float] = None
    carbs: Optional[float] = None
    fat: Optional[float] = None
    fiber: Optional[float] = None
    alcohol: Optional[float] = None
    saturated_fat: Optional[float] = None
    trans_fat: Optional[float] = None
    cholesterol: Optional[float] = None
    sodium: Optional[float] = None
    potassium: Optional[float] = None
    added_sugar: Optional[float] = None
    created_at: Optional[str] = None

    def convert_to_grams(self, amount: float, unit: str) -> Optional[float]:
        """Convert an amount in the given unit to grams.

        Returns None if conversion is not possible.
        """
        if unit not in VALID_UNITS:
            return None

        # Weight units - universal conversion
        if unit in WEIGHT_UNITS:
            return amount * WEIGHT_TO_GRAMS[unit]

        # Item-dependent units - need conversion factor
        if unit in ITEM_DEPENDENT_UNITS:
            if unit in self.unit_conversions:
                return amount * self.unit_conversions[unit]
            return None  # No conversion defined for this item

        # Volume units - cannot convert to grams without density
        if unit in VOLUME_UNITS:
            return None

        return None

    def add_unit_conversion(self, unit: str, grams_per_unit: float) -> None:
        """Add or update a unit conversion factor.

        Args:
            unit: The unit to convert from (must be in ITEM_DEPENDENT_UNITS)
            grams_per_unit: How many grams equal 1 of this unit
        """
        if unit not in ITEM_DEPENDENT_UNITS:
            raise ValueError(f"Unit '{unit}' is not item-dependent. "
                           f"Valid units: {ITEM_DEPENDENT_UNITS}")
        self.unit_conversions[unit] = grams_per_unit

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'id': self.id,
            'bar_code': self.bar_code,
            'name': self.name,
            'description': self.description,
            'unit_conversions': self.unit_conversions,
            'default_unit': self.default_unit,
            'calories': self.calories,
            'protein': self.protein,
            'carbs': self.carbs,
            'fat': self.fat,
            'fiber': self.fiber,
            'alcohol': self.alcohol,
            'saturated_fat': self.saturated_fat,
            'trans_fat': self.trans_fat,
            'cholesterol': self.cholesterol,
            'sodium': self.sodium,
            'potassium': self.potassium,
            'added_sugar': self.added_sugar,
            'created_at': self.created_at,
        }

    def get_nutrition(self, amount: float) -> dict:
        """Get scaled nutrition for given amount in default_unit."""
        return {
            'calories': int((self.calories or 0) * amount),
            'protein': (self.protein or 0) * amount,
            'carbs': (self.carbs or 0) * amount,
            'fat': (self.fat or 0) * amount,
            'fiber': (self.fiber or 0) * amount,
            'alcohol': (self.alcohol or 0) * amount,
            'saturated_fat': (self.saturated_fat or 0) * amount,
            'trans_fat': (self.trans_fat or 0) * amount,
            'cholesterol': (self.cholesterol or 0) * amount,
            'sodium': (self.sodium or 0) * amount,
            'potassium': (self.potassium or 0) * amount,
            'added_sugar': (self.added_sugar or 0) * amount,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Item':
        """Create Item from dictionary."""
        return cls(
            id=data.get('id'),
            bar_code=data.get('bar_code'),
            name=data['name'],
            description=data.get('description'),
            unit_conversions=data.get('unit_conversions', {}),
            default_unit=data.get('default_unit', 'item'),
            calories=data.get('calories'),
            protein=data.get('protein'),
            carbs=data.get('carbs'),
            fat=data.get('fat'),
            fiber=data.get('fiber'),
            alcohol=data.get('alcohol'),
            saturated_fat=data.get('saturated_fat'),
            trans_fat=data.get('trans_fat'),
            cholesterol=data.get('cholesterol'),
            sodium=data.get('sodium'),
            potassium=data.get('potassium'),
            added_sugar=data.get('added_sugar'),
        )
