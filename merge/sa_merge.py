#!/usr/bin/env python3
# 2024-2025 SPAPL

# Code for task vector generation borrowed from https://github.com/arcee-ai/mergekit

import os
import torch
import re
import numpy as np
import torch.nn as nn
import argparse
from transformers import AutoFeatureExtractor, AutoConfig, WhisperProcessor, WhisperForConditionalGeneration
import yaml

print('Test passed: Libraries imported successfully')


class TaskVector:
    """
    Task Vector Arithmetic for Model Merging.
    This class computes the difference between parameters of a finetuned model and a pretrained model,
    or initializes with a given task vector parameter dictionary.
    """

    def __init__(self, pretrained_model=None, finetuned_model=None, exclude_param_names_regex=None, task_vector_param_dict=None, in_ex_flag=True):
        self.task_vector_param_dict = task_vector_param_dict if task_vector_param_dict is not None else self._compute_task_vector(pretrained_model, finetuned_model, exclude_param_names_regex, in_ex_flag)

    def _compute_task_vector(self, pretrained_model, finetuned_model, exclude_param_names_regex, in_ex_flag):
        pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
        finetuned_param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
        
        param_names_to_merge = get_param_names_to_merge(list(pretrained_param_dict.keys()), exclude_param_names_regex, in_ex_flag)
        task_vector = {}
        
        with torch.no_grad():
            for param_name in param_names_to_merge:
                task_vector[param_name] = finetuned_param_dict[param_name] - pretrained_param_dict[param_name]
        return task_vector

    def __add__(self, other):
        assert isinstance(other, TaskVector), "Can only add another TaskVector!"
        new_task_vector_param_dict = {}
        
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                assert param_name in other.task_vector_param_dict, f"Parameter {param_name} not found in both TaskVectors!"
                new_task_vector_param_dict[param_name] = self.task_vector_param_dict[param_name] + other.task_vector_param_dict[param_name]
        
        return TaskVector(task_vector_param_dict=new_task_vector_param_dict)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        assert isinstance(other, (float, int)), "Can only multiply TaskVector by a float or int!"
        new_task_vector_param_dict = {param_name: other * self.task_vector_param_dict[param_name] for param_name in self.task_vector_param_dict}
        return TaskVector(task_vector_param_dict=new_task_vector_param_dict)

    def __rmul__(self, other):
        return self.__mul__(other)

    def combine_with_pretrained_model(self, pretrained_model, scaling_coefficient=1.0):
        pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
        merged_params = {}

        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                merged_params[param_name] = pretrained_param_dict[param_name] + scaling_coefficient * self.task_vector_param_dict[param_name]

        return merged_params

def get_param_names_to_merge(input_param_names, exclude_param_names_regex, in_ex_flag=True):
    """
    Get parameter names that should be merged, excluding those matching given patterns.
    """
    return [param_name for param_name in input_param_names if (all(not re.match(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex) if in_ex_flag else any(re.match(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex))]

def selective_attention_merging(merged_model, models_to_merge, exclude_param_names_regex, scaling_coefficient=1.0, in_ex_flag=True, lamda=1.0, alpha=1.0):
    """
    Selective Attention Merging of models using task vectors.
    """
    assert isinstance(scaling_coefficient, float), "Scaling coefficient must be a float!"

    models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model, exclude_param_names_regex=exclude_param_names_regex, in_ex_flag=in_ex_flag) for model in models_to_merge]

    with torch.no_grad():
        if lamda != 0.0:
            assert scaling_coefficient == 1.0, "Scaling coefficient must be 1.0 when using lamda (not 1 or 0)!"
            SA_scaling_coefficient = lamda ** alpha
            print(f'Selective Merging with lamda: {lamda}, alpha: {alpha}')
            merged_task_vector = SA_scaling_coefficient * models_to_merge_task_vectors[0] + (1 - SA_scaling_coefficient) * models_to_merge_task_vectors[1]
        else:
            assert scaling_coefficient == 1.0, "Scaling coefficient must be 1.0 when lamda = 0!"
            print(f'Only keeping parameters of model 1 with lamda: {lamda}')
            merged_task_vector = models_to_merge_task_vectors[1]

        merged_params = merged_task_vector.combine_with_pretrained_model(pretrained_model=merged_model, scaling_coefficient=scaling_coefficient)

    return merged_params

def copy_params_to_model(params, model):
    """
    Copy parameters from a dictionary to a model.
    """
    for param_name, param_value in model.named_parameters():
        if param_name in params:
            param_value.data.copy_(params[param_name])


def save_model_components(merged_model, processor, config, out_path):
    """
    Save the merged model, processor, and configuration to the specified output path.
    """
    os.makedirs(out_path, exist_ok=True)
    merged_model.save_pretrained(out_path)
    processor.save_pretrained(out_path)
    config.save_pretrained(out_path)
    print(f'Merged model and related components saved to {out_path}')


def sa_merge(model_class, model_paths, merge_configs, out_path):
    # Load the pretrained Whisper model
    merged_model = WhisperForConditionalGeneration.from_pretrained(model_class)
    merged_model.eval()

    # Load models and their components
    models_to_merge, configs, processors = [], [], []
    for model_path in model_paths:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'File {model_path} does not exist')

        feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        processor = WhisperProcessor.from_pretrained(os.path.join(model_path, ".."))
        processor.current_processor = feature_extractor
        processor.feature_extractor = feature_extractor
        model = WhisperForConditionalGeneration.from_pretrained(model_path, config=config).to("cpu")
        model.eval()

        models_to_merge.append(model)
        configs.append(config)
        processors.append(processor)


    # Define merging configurations; set different alpha for different layers
   

    # Execute the merging for each configuration
    exclude_para_names_regex_sum = [] # updated after each merge, used for the other layers (when in_ex_flag=True)
    for config in merge_configs:
        exclude_para_names_regex = config['include']
        exclude_para_names_regex_sum += exclude_para_names_regex
        merged_model_dict = selective_attention_merging(
            merged_model=merged_model, 
            models_to_merge=models_to_merge, 
            exclude_param_names_regex=exclude_para_names_regex, 
            scaling_coefficient=1.0, 
            in_ex_flag=False,
            lamda=config['lamda'], 
            alpha=config['alpha']
        )
        # # Print parameter names of the merged model for debugging and verification
        # for param_name, param_value in merged_model_dict.items():
        #     print('Merged model parameter name:', param_name)
        copy_params_to_model(merged_model_dict, merged_model)


    '''Selective Attention Merging Setup for Other Layers'''

    # Print the current exclude parameter regex list for debugging
    print('Exclude parameter names regex sum is:', exclude_para_names_regex_sum)

    # Set flags and parameters for the merging operation
    in_ex_flag = True  # Set to True to exclude parameters matching the regex, False to include them
    scaling_coefficient = 1.0  # Scaling factor for merging task vectors
    lamda = 0.0  # Merging weight; 0.0 keeps only MyST params, 1.0 applies task arithmetic

    # Perform the Selective Attention Merging based on the specified parameters
    merged_model_dict = selective_attention_merging(
        merged_model=merged_model, 
        models_to_merge=models_to_merge, 
        exclude_param_names_regex=exclude_para_names_regex_sum, 
        scaling_coefficient=scaling_coefficient, 
        in_ex_flag=in_ex_flag,
        lamda=lamda
    )

    # Print parameter names of the merged model for debugging and verification
    for param_name, param_value in merged_model_dict.items():
        print('Merged model parameter name:', param_name)
        # Uncomment the following lines if you want to see the parameter values and shapes
        # print(param_value)
        # print(param_value.shape)

    # Copy the merged parameters back to the original model
    copy_params_to_model(merged_model_dict, merged_model)

    # Save merged model components
    save_model_components(merged_model, processors[0], configs[0], out_path)

def main():
    parser = argparse.ArgumentParser(description='Process model paths and configs')
    parser.add_argument('--model_class', required=True, default= 'openai/whisper-small.en', help='Type of SFM model')
    parser.add_argument('--domain_ft_model', required=True, help='Low Resource domain finetuned model')
    parser.add_argument('--general_ft_model', required=True, help='General Speech finetuned model')
    parser.add_argument('--out_path', required=True, help='Output path')
    parser.add_argument('--config_path', required=True, help='Path to the YAML config file')

    

    args = parser.parse_args()
    model_class = args.model_class
    model_paths = [args.domain_ft_model, args.general_ft_model]
    out_path = args.out_path

    config_path = args.config_path

    with open(config_path, 'r') as file:
        merge_configs = yaml.safe_load(file)['merge_configs']

    sa_merge(model_class, model_paths, merge_configs, out_path)

    
if __name__ == '__main__':

    main()