"""Material Database - Manages lab material inventory"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from .property_mapper import PropertyMapper


class MaterialDatabase:
    """
    Manages a database of lab materials loaded from JSON format.
    Supports searching by properties using embedding-based property name mapping.
    """
    
    def __init__(
        self,
        materials: List[Dict[str, Any]] = None,
        property_mapper: Optional[PropertyMapper] = None
    ):
        """
        Initialize the material database.
        
        Args:
            materials: Optional list of material dictionaries to initialize with
            property_mapper: Optional PropertyMapper instance for property name mapping
        """
        self.materials = materials if materials is not None else []
        self._material_index = {}  # Index by material_id for fast lookup
        self.property_mapper = property_mapper
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild the material index."""
        self._material_index = {mat["material_id"]: mat for mat in self.materials}
    
    @classmethod
    def load_from_json(
        cls,
        path: str,
        property_mapper: Optional[PropertyMapper] = None
    ) -> "MaterialDatabase":
        """
        Load materials from a JSON file (internal_material_database.json format).
        
        Args:
            path: Path to the JSON file
            property_mapper: Optional PropertyMapper instance for property name mapping
            
        Returns:
            MaterialDatabase instance loaded with materials from file
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Material database file not found: {path}")
        
        with open(path_obj, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Normalize entries from JSON format
        materials = []
        for entry_id, entry_data in data.items():
            # Extract properties from "extracted" field (prefer over llm_response_json.extracted)
            properties = {}
            extracted = entry_data.get("extracted", {})
            if not extracted:
                # Fallback to llm_response_json.extracted
                llm_extracted = entry_data.get("llm_response_json", {}).get("extracted", {})
                if llm_extracted:
                    extracted = llm_extracted
            
            # Use extracted properties as-is (will be mapped during search)
            properties = extracted.copy() if extracted else {}
            
            # Create normalized material entry
            material = {
                "material_id": entry_data.get("id", entry_id),
                "material_name": entry_data.get("name", ""),
                "supplier": entry_data.get("supplier", ""),
                "properties": properties,  # Raw properties from extracted field
                "source_files": entry_data.get("source_files", []),
                "raw_data": entry_data  # Keep original for reference
            }
            materials.append(material)
        
        return cls(materials=materials, property_mapper=property_mapper)
    
    def set_property_mapper(self, property_mapper: PropertyMapper):
        """Set the property mapper for this database."""
        self.property_mapper = property_mapper
    
    def add_material(self, material_dict: Dict[str, Any]) -> str:
        """
        Add a new material to the database.
        
        Args:
            material_dict: Dictionary containing material information with required keys:
                - material_id: Unique identifier
                - material_name: Common name
                - properties: Dict of property names to values
                
        Returns:
            The material_id of the added material
        """
        required_keys = ["material_id", "material_name", "properties"]
        for key in required_keys:
            if key not in material_dict:
                raise ValueError(f"Material dictionary must contain '{key}' key")
        
        material_id = material_dict["material_id"]
        if material_id in self._material_index:
            raise ValueError(f"Material with ID '{material_id}' already exists")
        
        self.materials.append(material_dict)
        self._material_index[material_id] = material_dict
        
        return material_id
    
    def get_material(self, material_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a material by its ID.
        
        Args:
            material_id: The material ID to look up
            
        Returns:
            Material dictionary if found, None otherwise
        """
        return self._material_index.get(material_id)
    
    def search_by_properties(
        self,
        properties_dict: Dict[str, Any],
        match_all: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for materials matching given properties using embedding-based property name mapping.
        
        Args:
            properties_dict: Dictionary of target property names to target values/ranges
            match_all: If True, material must match all properties; if False, matches any property
            
        Returns:
            List of material dictionaries that match the criteria
        """
        if not self.property_mapper:
            raise ValueError("PropertyMapper must be set before searching. Use set_property_mapper() or pass it during initialization.")
        
        target_properties = list(properties_dict.keys())
        matches = []
        
        for material in self.materials:
            material_props = material.get("properties", {})
            matched_props = []
            
            # Map each database property to target properties using embeddings
            db_prop_to_target = {}  # Maps DB property name -> target property name
            for db_prop_name in material_props.keys():
                mapped_target = self.property_mapper.map_property_name(
                    db_prop_name,
                    target_properties
                )
                if mapped_target:
                    db_prop_to_target[db_prop_name] = mapped_target
            
            # Check each target property
            for target_prop, target_value in properties_dict.items():
                # Find DB properties that map to this target property
                matching_db_props = [
                    db_prop for db_prop, mapped_target in db_prop_to_target.items()
                    if mapped_target == target_prop
                ]
                
                # Check if any matching DB property has a value that matches target
                for db_prop_name in matching_db_props:
                    db_value = material_props[db_prop_name]
                    if self.property_mapper.compare_property_values(db_value, target_value):
                        matched_props.append(target_prop)
                        break  # Found a match for this target property
            
            # Check if material matches criteria
            if match_all:
                if len(matched_props) == len(properties_dict):
                    matches.append(material)
            else:
                if len(matched_props) > 0:
                    matches.append(material)
        
        return matches
    
    def get_all_materials(self) -> List[Dict[str, Any]]:
        """
        Get all materials in the database.
        
        Returns:
            List of all material dictionaries
        """
        return self.materials.copy()
    
    def __len__(self) -> int:
        """Return the number of materials in the database."""
        return len(self.materials)
    
    def __repr__(self) -> str:
        """String representation of the database."""
        return f"MaterialDatabase({len(self.materials)} materials)"
