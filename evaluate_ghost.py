"""GHOST evaluation script with GhostConsistencyScore metric."""

import json
import argparse
from typing import List, Dict, Any, Tuple
from ghost_consistency_score import calculate_gcs, calculate_combined_gcs, calculate_confusion_matrix, calculate_accuracy
from utils import post_process_prediction, validate_prediction_format

def load_predictions(pred_path: str) -> List[Dict[str, Any]]:
    """Load predictions from JSON file."""
    with open(pred_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not validate_prediction_format(data):
        raise ValueError(f"Invalid prediction format in {pred_path}")
    return data

def restructure_predictions(data: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Convert flat prediction list to hierarchical structure."""
    restructured_data = {}
    stats = {
        'total_questions': 0,
        'unique_object_ids': set(),
        'counts': {'objects': 0, 'attributes': 0, 'relations': 0},
        'prediction_types': {'true': 0, 'false': 0, 'unknown': 0}
    }
    
    for item in data:
        stats['total_questions'] += 1
        parts = item['question_id'].split('_')
        image_id = parts[0]
        obj_id = f"{parts[0]}_{parts[1]}"
        question_type = '_'.join(parts[2:])
        stats['unique_object_ids'].add(obj_id)
        
        if image_id not in restructured_data:
            restructured_data[image_id] = {}
        if obj_id not in restructured_data[image_id]:
            restructured_data[image_id][obj_id] = {}
        
        processed_prediction = post_process_prediction(item['prediction'])
        stats['prediction_types'][processed_prediction] += 1
        
        if question_type.startswith('attr'):
            category_key = 'attr'
            stats['counts']['attributes'] += 1
        elif question_type.startswith('rel'):
            category_key = 'rel'
            stats['counts']['relations'] += 1
        else:
            category_key = 'questions'
            stats['counts']['objects'] += 1
        
        if category_key not in restructured_data[image_id][obj_id]:
            restructured_data[image_id][obj_id][category_key] = {}
        
        restructured_data[image_id][obj_id][category_key][question_type] = {
            'text': item['text'],
            'label': item['label'],
            'prediction': processed_prediction
        }
    
    stats['unique_object_ids'] = len(stats['unique_object_ids'])
    return restructured_data, stats

def evaluate(pred_path: str, minimal: bool = False) -> Dict[str, Any]:
    """Main evaluation function."""
    predictions = load_predictions(pred_path)
    restructured_data, stats = restructure_predictions(predictions)
    
    objects_result = calculate_gcs(restructured_data, 'objects')
    attributes_result = calculate_gcs(restructured_data, 'attributes')
    relations_result = calculate_gcs(restructured_data, 'relations')
    combined_result = calculate_combined_gcs(restructured_data)
    combined_cm = calculate_confusion_matrix(restructured_data)
    
    results = {
        'objects_gcs': objects_result['mean_score'],
        'attributes_gcs': attributes_result['mean_score'],
        'relations_gcs': relations_result['mean_score'],
        'combined_gcs': combined_result['mean_score'],
        'accuracy': calculate_accuracy(combined_cm),
        'confusion_matrix': combined_cm,
        'statistics': stats
    }
    
    if minimal:
        print(f"  Objects GCS:      {results['objects_gcs']:.2f}%")
        print(f"  Attributes GCS:   {results['attributes_gcs']:.2f}%")
        print(f"  Relations GCS:    {results['relations_gcs']:.2f}%")
    else:
        print("\n" + "=" * 60)
        print("GHOST EVALUATION RESULTS")
        print("=" * 60)
        print(f"\nDataset Statistics:")
        print(f"  Total Questions:      {stats['total_questions']}")
        print(f"  Unique Objects:       {stats['unique_object_ids']}")
        print(f"  Object Questions:     {stats['counts']['objects']}")
        print(f"  Attribute Questions:  {stats['counts']['attributes']}")
        print(f"  Relation Questions:   {stats['counts']['relations']}")
        print(f"\nPrediction Distribution:")
        print(f"  True:    {stats['prediction_types']['true']}")
        print(f"  False:   {stats['prediction_types']['false']}")
        print(f"  Unknown: {stats['prediction_types']['unknown']}")
        print("\n" + "-" * 60)
        print("GhostConsistencyScore (GCS) Results:")
        print("-" * 60)
        print(f"\n  Objects GCS:      {results['objects_gcs']:.2f}%")
        print(f"  Attributes GCS:   {results['attributes_gcs']:.2f}%")
        print(f"  Relations GCS:    {results['relations_gcs']:.2f}%")
        print(f"\n  Combined GCS:     {results['combined_gcs']:.2f}%")
        print("\n" + "-" * 60)
        print("Confusion Matrix (Combined):")
        print("-" * 60)
        print(f"  True Positives:   {combined_cm['TP']}")
        print(f"  True Negatives:   {combined_cm['TN']}")
        print(f"  False Positives:  {combined_cm['FP']}")
        print(f"  False Negatives:  {combined_cm['FN']}")
        print(f"\n  Accuracy:         {results['accuracy']:.2f}%")
        print("=" * 60 + "\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions using GhostConsistencyScore")
    parser.add_argument("--pred-path", type=str, required=True, help="Path to prediction JSON file")
    parser.add_argument("--minimal", action="store_true", help="Show only GCS scores")
    args = parser.parse_args()
    
    evaluate(args.pred_path, minimal=args.minimal)

if __name__ == "__main__":
    main()

