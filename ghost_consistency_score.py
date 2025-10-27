"""GhostConsistencyScore metric calculation."""

from typing import Dict, List

def sum_i(i: int) -> float:
    """Compute sum of 1/(2^j) for j from 0 to i-1."""
    return sum(1.0 / (2 ** j) for j in range(i))

def calculate_gcs(predictions: Dict, category: str) -> Dict:
    """Calculate GCS for a specific category (objects/attributes/relations)."""
    category_mapping = {
        'objects': 'questions',
        'attributes': 'attr',
        'relations': 'rel'
    }
    
    data_key = category_mapping.get(category)
    if not data_key:
        raise ValueError(f"Invalid category: {category}")
    
    scores = []
    for image_data in predictions.values():
        for obj_data in image_data.values():
            questions = obj_data.get(data_key, {})
            if not questions:
                continue
            
            num_incorrect = sum(
                1 for d in questions.values()
                if not ((d['label'] == 'yes' and d['prediction'] == 'true') or
                       (d['label'] == 'no' and d['prediction'] == 'false'))
            )
            
            total_possible = sum_i(len(questions))
            incorrect_sum = sum_i(num_incorrect)
            score = 1 - (incorrect_sum / total_possible) if total_possible > 0 else 1.0
            scores.append(score)
    
    return {
        'scores': scores,
        'mean_score': (sum(scores) / len(scores) * 100) if scores else 0.0,
        'num_objects': len(scores)
    }

def calculate_combined_gcs(predictions: Dict) -> Dict:
    """Calculate combined GCS across all question types."""
    category_mapping = {
        'objects': 'questions',
        'attributes': 'attr',
        'relations': 'rel'
    }
    
    scores = []
    for image_data in predictions.values():
        for obj_data in image_data.values():
            total_questions = 0
            total_incorrect = 0
            
            for data_key in category_mapping.values():
                questions = obj_data.get(data_key, {})
                total_questions += len(questions)
                
                for d in questions.values():
                    if not ((d['label'] == 'yes' and d['prediction'] == 'true') or
                           (d['label'] == 'no' and d['prediction'] == 'false')):
                        total_incorrect += 1
            
            if total_questions > 0:
                score = 1 - (sum_i(total_incorrect) / sum_i(total_questions))
            else:
                score = 1.0
            scores.append(score)
    
    return {
        'scores': scores,
        'mean_score': (sum(scores) / len(scores) * 100) if scores else 0.0,
        'num_objects': len(scores)
    }
