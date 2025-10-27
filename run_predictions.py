"""GHOST prediction generation with checkpoint/resume support."""

import json
import argparse
import os
import re
from typing import Dict, List, Set
from tqdm import tqdm
from utils import validate_dataset_format, get_image_filename_from_key, get_object_id_from_key, get_label_from_key

def load_dataset(json_path: str) -> Dict[str, str]:
    """Load GHOST dataset from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not validate_dataset_format(data):
        raise ValueError(f"Invalid dataset format in {json_path}")
    return data

def parse_and_convert(text: str) -> str:
    """Convert GHOST statement format to natural language."""
    object_match = re.match(r"A (.*) is present in the image\.", text)
    if object_match:
        return f"There is a {object_match.group(1)} in the image."
    
    attribute_match = re.match(r"The (.*?) of the (.*?) present in the image is (.*?)\.", text)
    if attribute_match:
        attr, obj, val = attribute_match.groups()
        return f"In the image, the {attr} of the {obj} is {val}."
    
    relation_match = re.match(r"The (.*?) between the (.*?) and (.*?) is that the (.*?)\.", text)
    if relation_match:
        rel, obj1, obj2, desc = relation_match.groups()
        if rel == "spatial":
            return f"In the image, the spatial relationship between the {obj1} and {obj2} is: {desc}."
        return f"In the image, the {rel} connecting the {obj1} and the {obj2} is that {desc}."
    
    return text

def format_prompt(text: str) -> str:
    """Format statement as True/False question."""
    return f"Is the following statement about the image true or false: '{text}'? Please respond with only 'True' or 'False'."

def get_vlm_prediction(model, image_path: str, prompt: str) -> str:
    """Get prediction from local VLM model."""
    prediction = model.generate([image_path, prompt], dataset='Y/N')
    return prediction.strip().lower()

def get_api_prediction(model_name: str, image_path: str, prompt: str, api_key: str = None) -> str:
    """Get prediction from API model (placeholder - implement as needed)."""
    raise NotImplementedError(
        f"API prediction for {model_name} not implemented. "
        "Implement API call logic in get_api_prediction()."
    )

def format_output(question_id: str, object_id: str, image_file: str, prompt: str, 
                 correct_answer: str, model_name: str, prediction: str) -> Dict:
    """Format prediction output."""
    return {
        "question_id": question_id,
        "object_id": object_id,
        "image": image_file,
        "text": prompt,
        "label": correct_answer,
        "model_name": model_name,
        "prediction": prediction
    }

def load_checkpoint(output_path: str) -> tuple[List[Dict], Set[str]]:
    """Load existing predictions and get completed question IDs."""
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        completed_ids = {r['question_id'] for r in results}
        return results, completed_ids
    return [], set()

def save_checkpoint(results: List[Dict], output_path: str):
    """Save predictions to file (atomic write)."""
    temp_path = output_path + '.tmp'
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    os.replace(temp_path, output_path)

def run_vlm_predictions(data_path: str, image_dir: str, model_name: str, 
                       output_path: str, checkpoint_every: int = 10, resume: bool = True):
    """Run predictions using local VLM model with checkpoint support."""
    try:
        from vlmeval.config import supported_VLM
    except ImportError:
        raise ImportError("VLMEvalKit not found. Install it to use local VLM models.")
    
    dataset = load_dataset(data_path)
    print(f"Loaded {len(dataset)} questions")
    
    if resume:
        results, completed_ids = load_checkpoint(output_path)
        if completed_ids:
            print(f"Resuming from checkpoint: {len(completed_ids)} questions already completed")
    else:
        results, completed_ids = [], set()
    
    if model_name not in supported_VLM:
        raise ValueError(f"Model {model_name} not found in VLMEvalKit")
    
    model = supported_VLM[model_name]()
    
    questions_to_process = [(qid, text) for qid, text in dataset.items() if qid not in completed_ids]
    print(f"Processing {len(questions_to_process)} remaining questions...")
    
    for idx, (question_id, prompt_text) in enumerate(tqdm(questions_to_process, desc="Processing"), 1):
        image_file = get_image_filename_from_key(question_id)
        object_id = get_object_id_from_key(question_id)
        correct_answer = get_label_from_key(question_id)
        
        image_path = os.path.join(image_dir, image_file)
        converted_prompt = parse_and_convert(prompt_text)
        full_prompt = format_prompt(converted_prompt)
        
        prediction = get_vlm_prediction(model, image_path, full_prompt)
        
        result = format_output(question_id, object_id, image_file, prompt_text, 
                              correct_answer, model_name, prediction)
        results.append(result)
        
        if idx % checkpoint_every == 0:
            save_checkpoint(results, output_path)
    
    save_checkpoint(results, output_path)
    print(f"\nResults saved to: {output_path}")

def run_api_predictions(data_path: str, image_dir: str, model_name: str, 
                       output_path: str, api_key: str = None, checkpoint_every: int = 10, resume: bool = True):
    """Run predictions using API model with checkpoint support."""
    dataset = load_dataset(data_path)
    print(f"Loaded {len(dataset)} questions")
    
    if resume:
        results, completed_ids = load_checkpoint(output_path)
        if completed_ids:
            print(f"Resuming from checkpoint: {len(completed_ids)} questions already completed")
    else:
        results, completed_ids = [], set()
    
    questions_to_process = [(qid, text) for qid, text in dataset.items() if qid not in completed_ids]
    print(f"Processing {len(questions_to_process)} remaining questions...")
    
    for idx, (question_id, prompt_text) in enumerate(tqdm(questions_to_process, desc="Processing"), 1):
        image_file = get_image_filename_from_key(question_id)
        object_id = get_object_id_from_key(question_id)
        correct_answer = get_label_from_key(question_id)
        
        image_path = os.path.join(image_dir, image_file)
        converted_prompt = parse_and_convert(prompt_text)
        full_prompt = format_prompt(converted_prompt)
        
        prediction = get_api_prediction(model_name, image_path, full_prompt, api_key)
        
        result = format_output(question_id, object_id, image_file, prompt_text, 
                              correct_answer, model_name, prediction)
        results.append(result)
        
        if idx % checkpoint_every == 0:
            save_checkpoint(results, output_path)
    
    save_checkpoint(results, output_path)
    print(f"\nResults saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run model predictions on GHOST dataset")
    parser.add_argument("--data-path", type=str, required=True, help="Path to dataset JSON")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--model-name", type=str, required=True, help="Model name")
    parser.add_argument("--model-type", type=str, choices=['vlm', 'api'], required=True, help="Model type")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save predictions")
    parser.add_argument("--api-key", type=str, default=None, help="API key (for API models)")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save checkpoint every N questions")
    parser.add_argument("--no-resume", action="store_true", help="Start from scratch (ignore checkpoints)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")
    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    
    resume = not args.no_resume
    
    if args.model_type == 'vlm':
        run_vlm_predictions(args.data_path, args.image_dir, args.model_name, 
                           args.output_path, args.checkpoint_every, resume)
    else:
        run_api_predictions(args.data_path, args.image_dir, args.model_name, 
                           args.output_path, args.api_key, args.checkpoint_every, resume)

if __name__ == "__main__":
    main()

