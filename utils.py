"""Utility functions for GHOST dataset processing."""

from typing import Dict, List, Any

def parse_question_key(key: str) -> Dict[str, str]:
    """Extract metadata from question key."""
    parts = key.split('_')
    return {
        'image_id': parts[0],
        'object_id': f"{parts[0]}_{parts[1]}",
        'question_type': '_'.join(parts[2:]),
        'label_type': 'pos' if key.endswith('pos') else 'neg',
        'category': get_category_from_key(key)
    }

def get_category_from_key(key: str) -> str:
    """Determine category from question key."""
    if 'attr' in key:
        return 'attributes'
    elif 'rel' in key:
        return 'relations'
    return 'objects'

def validate_dataset_format(data: Dict[str, str]) -> bool:
    """Validate dataset format."""
    if not isinstance(data, dict) or len(data) == 0:
        return False
    sample_keys = list(data.keys())[:10]
    return all(len(k.split('_')) >= 3 and isinstance(data[k], str) for k in sample_keys)

def validate_prediction_format(data: List[Dict[str, Any]]) -> bool:
    """Validate prediction format."""
    if not isinstance(data, list) or len(data) == 0:
        return False
    required = ['question_id', 'object_id', 'image', 'text', 'label', 'model_name', 'prediction']
    return all(isinstance(e, dict) and all(f in e for f in required) for e in data[:10])

def post_process_prediction(prediction: str) -> str:
    """Normalize prediction to true/false/unknown."""
    pred_lower = prediction.strip().rstrip('.').lower()
    
    if pred_lower in ['true', 'yes']:
        return 'true'
    elif pred_lower in ['false', 'no']:
        return 'false'
    
    first_word = pred_lower.split()[0] if pred_lower else ''
    if first_word in ['true', 'yes']:
        return 'true'
    elif first_word in ['false', 'no']:
        return 'false'
    
    if any(w in pred_lower for w in ['true', 'yes']):
        return 'true'
    elif any(w in pred_lower for w in ['false', 'no']):
        return 'false'
    
    return 'unknown'

def get_image_filename_from_key(key: str) -> str:
    """Extract image filename from question key."""
    return f"{key.split('_')[0]}.jpg"

def get_object_id_from_key(key: str) -> str:
    """Extract object ID from question key."""
    parts = key.split('_')
    return f"{parts[0]}_{parts[1]}"

def get_label_from_key(key: str) -> str:
    """Determine label from question key."""
    return 'yes' if key.endswith('pos') else 'no'

