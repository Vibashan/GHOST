"""GHOST evaluation script with GhostConsistencyScore metric."""

import json
import argparse
from typing import List, Dict, Any, Tuple
from ghost_consistency_score import calculate_gcs, calculate_combined_gcs
from utils import post_process_prediction, validate_prediction_format

def load_predictions(pred_path: str) -> List[Dict[str, Any]]:
    """Load predictions from JSON file."""
    with open(pred_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not validate_prediction_format(data):
        raise ValueError(f"Invalid prediction format in {pred_path}")
    return data

def restructure_predictions(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert flat prediction list to hierarchical structure."""
    restructured_data = {}
    
    for item in data:
        parts = item['question_id'].split('_')
        image_id = parts[0]
        obj_id = f"{parts[0]}_{parts[1]}"
        question_type = '_'.join(parts[2:])
        
        if image_id not in restructured_data:
            restructured_data[image_id] = {}
        if obj_id not in restructured_data[image_id]:
            restructured_data[image_id][obj_id] = {}
        
        processed_prediction = post_process_prediction(item['prediction'])
        
        if question_type.startswith('attr'):
            category_key = 'attr'
        elif question_type.startswith('rel'):
            category_key = 'rel'
        else:
            category_key = 'questions'
        
        if category_key not in restructured_data[image_id][obj_id]:
            restructured_data[image_id][obj_id][category_key] = {}
        
        restructured_data[image_id][obj_id][category_key][question_type] = {
            'text': item['text'],
            'label': item['label'],
            'prediction': processed_prediction
        }
    
    return restructured_data

def evaluate(pred_path: str) -> Dict[str, Any]:
    """Main evaluation function."""
    predictions = load_predictions(pred_path)
    restructured_data = restructure_predictions(predictions)
    
    objects_result = calculate_gcs(restructured_data, 'objects')
    attributes_result = calculate_gcs(restructured_data, 'attributes')
    relations_result = calculate_gcs(restructured_data, 'relations')
    combined_result = calculate_combined_gcs(restructured_data)
    
    results = {
        'objects_gcs': objects_result['mean_score'],
        'attributes_gcs': attributes_result['mean_score'],
        'relations_gcs': relations_result['mean_score'],
        'combined_gcs': combined_result['mean_score']
    }
    
    print(f"  Objects GCS:      {results['objects_gcs']:.2f}%")
    print(f"  Attributes GCS:   {results['attributes_gcs']:.2f}%")
    print(f"  Relations GCS:    {results['relations_gcs']:.2f}%")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions using GhostConsistencyScore")
    parser.add_argument("--pred-path", type=str, required=True, help="Path to prediction JSON file")
    args = parser.parse_args()
    
    evaluate(args.pred_path)

if __name__ == "__main__":
    main()
