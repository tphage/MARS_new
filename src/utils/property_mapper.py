"""Property Mapper - Maps property names using embeddings and normalizes property values"""

import re
import numpy as np
from typing import List, Dict, Any, Optional


class PropertyMapper:
    """
    Maps property names from database to target properties using embedding similarity,
    and normalizes property values (units, ranges, qualitative descriptions).
    """
    
    def __init__(
        self,
        embedding_model,
        embedding_tokenizer: str = "",
        similarity_threshold: float = None
    ):
        """
        Initialize the property mapper.
        
        Args:
            embedding_model: SentenceTransformer model for generating embeddings
            embedding_tokenizer: Tokenizer (not used but kept for compatibility)
            similarity_threshold: Minimum similarity threshold for property mapping (default: None, uses config value)
        """
        # Load config
        from ..config import load_config
        config = load_config()
        utils_config = config.get("utils", {}).get("property_mapper", {})
        
        # Use config default if not provided
        if similarity_threshold is None:
            similarity_threshold = utils_config.get("similarity_threshold", 0.8)
        
        self.embedding_model = embedding_model
        self.embedding_tokenizer = embedding_tokenizer
        self.similarity_threshold = similarity_threshold
        self.numeric_tolerance = utils_config.get("numeric_tolerance", 0.01)
        self.max_unit_string_length = utils_config.get("max_unit_string_length", 20)
    
    def map_property_name(
        self,
        property_name: str,
        target_properties: List[str]
    ) -> Optional[str]:
        """
        Map a property name from the database to a target property name using embedding similarity.
        
        Args:
            property_name: Property name from database (e.g., "Deflection Temperature Under Load")
            target_properties: List of target property names (e.g., ["thermal_stability", "chemical_resistance"])
            
        Returns:
            Best matching target property name if similarity >= threshold, None otherwise
        """
        if not target_properties:
            return None
        
        # Generate embeddings
        try:
            # Combine property_name with all target properties for batch encoding
            all_texts = [property_name] + target_properties
            embeddings = self.embedding_model.encode(all_texts, convert_to_numpy=True)
            
            property_embedding = embeddings[0]
            target_embeddings = embeddings[1:]
            
            # Calculate cosine similarities
            similarities = []
            for target_emb in target_embeddings:
                # Cosine similarity
                similarity = np.dot(property_embedding, target_emb) / (
                    np.linalg.norm(property_embedding) * np.linalg.norm(target_emb)
                )
                similarities.append(float(similarity))
            
            # Find best match
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            if best_similarity >= self.similarity_threshold:
                return target_properties[best_idx]
            else:
                return None
                
        except Exception as e:
            # If embedding fails, return None
            return None
    
    def normalize_property_value(self, value: str) -> Dict[str, Any]:
        """
        Normalize a property value string to structured format.
        
        Handles:
        - Values with units (e.g., "1.30 g/cm³")
        - Ranges (e.g., "0.20 to 0.30 %")
        - Qualitative values (e.g., "excellent", "high")
        - Single numeric values (e.g., "3600")
        
        Args:
            value: Property value string
            
        Returns:
            Dict with keys:
                - "value": float (single value or midpoint of range)
                - "unit": str (extracted unit if present)
                - "range": [min, max] or None
                - "qualitative": str (qualitative description if present)
                - "original": str (original value string)
        """
        original = str(value).strip()
        result = {
            "value": None,
            "unit": None,
            "range": None,
            "qualitative": None,
            "original": original
        }
        
        if not original or original.lower() in ["--", "n/a", "none", "unknown"]:
            return result
        
        # Check for qualitative values first
        qualitative_keywords = [
            "excellent", "very_high", "very good", "outstanding",
            "good", "moderate", "acceptable", "high", "strong",
            "yes", "true", "present", "no", "false", "absent",
            "no break", "v-0", "v0"
        ]
        
        value_lower = original.lower()
        for qual in qualitative_keywords:
            if qual in value_lower:
                result["qualitative"] = original
                # Try to extract numeric value if present
                numbers = re.findall(r'[\d.]+', original)
                if numbers:
                    try:
                        result["value"] = float(numbers[0])
                    except (ValueError, IndexError):
                        pass
                return result
        
        # Try to extract range (e.g., "0.20 to 0.30", "1.28 to 1.30")
        range_pattern = r'([\d.]+)\s*(?:to|-)\s*([\d.]+)'
        range_match = re.search(range_pattern, original, re.IGNORECASE)
        
        if range_match:
            try:
                min_val = float(range_match.group(1))
                max_val = float(range_match.group(2))
                result["range"] = [min_val, max_val]
                result["value"] = (min_val + max_val) / 2.0  # Midpoint
            except (ValueError, TypeError):
                pass
        
        # Extract numeric value(s)
        numbers = re.findall(r'[\d.]+', original)
        if numbers and result["value"] is None:
            try:
                # Use first number if no range found
                result["value"] = float(numbers[0])
            except (ValueError, IndexError):
                pass
        
        # Extract unit (common units)
        unit_patterns = [
            (r'([\d.]+)\s*(g/cm³|g/cm3|kg/m³|kg/m3)', 'density'),
            (r'([\d.]+)\s*(°c|°c|°c|°c|celsius)', 'temperature'),
            (r'([\d.]+)\s*(mpa|pa|gpa|psi)', 'pressure'),
            (r'([\d.]+)\s*(%|percent)', 'percentage'),
            (r'([\d.]+)\s*(kv/mm|v/mm)', 'voltage'),
            (r'([\d.]+)\s*(kj/m²|j/m|j/m²)', 'energy'),
            (r'([\d.]+)\s*(w/m/k|w/m·k)', 'thermal_conductivity'),
        ]
        
        for pattern, unit_type in unit_patterns:
            match = re.search(pattern, original, re.IGNORECASE)
            if match:
                result["unit"] = match.group(2).lower()
                break
        
        # If no unit found but there's text after number, try to extract it
        if result["unit"] is None and result["value"] is not None:
            # Look for text after the number
            after_number = re.sub(r'^[\d.\s-]+', '', original).strip()
            if after_number and len(after_number) < self.max_unit_string_length:
                result["unit"] = after_number
        
        return result
    
    def compare_property_values(
        self,
        db_value: str,
        target_value: Optional[str]
    ) -> bool:
        """
        Compare a database property value against a target value.
        
        Args:
            db_value: Property value from database
            target_value: Target value to match against (None means any value acceptable)
            
        Returns:
            True if values match, False otherwise
        """
        if target_value is None:
            # Any value is acceptable
            return True
        
        # Normalize both values
        db_norm = self.normalize_property_value(db_value)
        target_norm = self.normalize_property_value(str(target_value))
        
        # If both have qualitative values, compare them
        if db_norm["qualitative"] and target_norm["qualitative"]:
            # Use synonym matching
            db_qual = db_norm["qualitative"].lower()
            target_qual = target_norm["qualitative"].lower()
            
            synonyms = {
                "excellent": ["very_high", "very good", "outstanding", "high"],
                "good": ["moderate", "acceptable", "high"],
                "high": ["strong", "good", "excellent"],
                "yes": ["true", "1", "present"],
                "no": ["false", "0", "absent"],
                "v-0": ["v0", "v-0", "flame_retardant"],
            }
            
            # Check direct match
            if db_qual == target_qual:
                return True
            
            # Check synonyms
            for key, values in synonyms.items():
                if db_qual == key and target_qual in values:
                    return True
                if target_qual == key and db_qual in values:
                    return True
            
            return False
        
        # If both have numeric values, compare them
        if db_norm["value"] is not None and target_norm["value"] is not None:
            # Handle ranges
            if db_norm["range"]:
                db_min, db_max = db_norm["range"]
                target_val = target_norm["value"]
                # Check if target value is within range
                if db_min <= target_val <= db_max:
                    return True
            
            if target_norm["range"]:
                target_min, target_max = target_norm["range"]
                db_val = db_norm["value"]
                # Check if db value is within target range
                if target_min <= db_val <= target_max:
                    return True
            
            # Direct numeric comparison (with tolerance for floating point)
            if abs(db_norm["value"] - target_norm["value"]) < self.numeric_tolerance:
                return True
            
            # Check if target is a comparison (e.g., ">150C")
            target_str = str(target_value).lower()
            if ">" in target_str or "greater" in target_str:
                if db_norm["value"] > target_norm["value"]:
                    return True
            elif "<" in target_str or "less" in target_str:
                if db_norm["value"] < target_norm["value"]:
                    return True
        
        # Fallback: string matching
        db_str = str(db_value).lower().strip()
        target_str = str(target_value).lower().strip()
        
        if db_str == target_str:
            return True
        
        if target_str in db_str or db_str in target_str:
            return True
        
        return False
