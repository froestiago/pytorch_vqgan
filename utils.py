import os
import yaml
import argparse

from PIL import Image
from datetime import datetime


def str2bool(value):
    """
    Convert a string representation of truthy or falsy values to a boolean.

    Parameters:
    value (str or bool): The input value to convert. Can be a boolean or a string representing 
                         a boolean value ('true', 't', 'yes', 'y', '1' for True; 
                         'false', 'f', 'no', 'n', '0' for False). Case-insensitive.

    Returns:
    bool: The corresponding boolean value.

    Raises:
    argparse.ArgumentTypeError: If the input string does not match expected boolean values.
    """

    
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif value.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected. Got {value}")
    

def denormalize_image(image):
    """
    Denormalize image from [-1, 1] range back to [0, 1] range for saving.
    """
    return (image + 1.0) / 2.0
    

def norm_image(image):
    """
    Normalize an image to the range [0, 1].

    Parameters:
    image (numpy.ndarray or torch.Tensor): The input image array.

    Returns:
    numpy.ndarray or torch.Tensor: The normalized image, where pixel values are scaled 
                                   to the range [0, 1].

    Notes:
    - This function performs min-max normalization using the formula:
      (image - image.min()) / (image.max() - image.min()).
    - If `image.min() == image.max()`, this will result in division by zero.
    """


    norm_image = (image - image.min()) / (image.max() - image.min())
    return norm_image


def create_output_dir(output_dir):
    """
    Create an output directory structure for storing generated files.

    This function ensures that the following directory structure exists:
    ./outputs/
    ├── <output_dir>/
        ├── generated/
        ├── validation/
        ├── checkpoints/

    If any of these directories do not exist, they are created.

    Parameters:
    output_dir (str): The name of the subdirectory within './outputs/' to be created.

    Notes:
    - If the directories already exist, no action is taken.
    - Ensure that the function has appropriate write permissions for creating directories.
    """


    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')
    
    subdirs = ['generated', 'validation', 'checkpoints']
    for subdir in subdirs:
        path = f'./outputs/{output_dir}/{subdir}/'
        if not os.path.exists(path):
            os.makedirs(path)


def save_config_file_copy_at_output(args):
    """
    Save a copy of the configuration file in the output directory with a timestamp.

    This function takes an argparse.Namespace object, converts it to a dictionary, and saves 
    it as a YAML file inside the specified output directory. The filename includes a timestamp 
    to ensure uniqueness.

    Parameters:
    args (argparse.Namespace): The parsed command-line arguments, expected to have an 
                               'output_dir' attribute specifying the output directory.

    Notes:
    - The YAML file is saved in './outputs/{args.output_dir}/' with the format 'config_YYYY-MM-DD_HH-MM-SS.yaml'.
    - The function assumes that the './outputs/{args.output_dir}/' directory already exists.
    - Uses `yaml.safe_dump()` to ensure safe YAML serialization.
    """


    # Convert Namespace to dictionary
    config_dict = vars(args)

    # Get the current date and time
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Create the output file name with timestamp
    output_file = f"./outputs/{args.output_dir}/config_{timestamp}.yaml"

    # Save the config dictionary to the YAML file
    with open(output_file, 'w') as file:
        yaml.safe_dump(config_dict, file)