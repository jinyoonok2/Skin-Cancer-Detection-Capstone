import argparse
from model_handler import ModelHandler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Active Learning Loop')

    # Set default=None for model_path
    parser.add_argument('--data_path', type=str, help='path to data yaml')
    parser.add_argument('--model_path', type=str, default=None, help='path to model (optional)')

    # Arguments for classify_data method
    parser.add_argument('--img_path', type=str, default=None, help='path to image directory (optional)')
    parser.add_argument('--table_path', type=str, default=None, help='path to table csv (optional)')
    parser.add_argument('--dataset_version', type=str, default=None, help='version of the dataset (optional)')

    args = parser.parse_args()

    # Initialize ModelHandler with or without model_path
    # if args.model_path:
    #     model_handler = ModelHandler(args.data_path, args.model_path)
    # else:
    #     model_handler = ModelHandler(args.data_path)
    #
    # # Optionally, call classify_data if paths and version are provided
    # if args.img_path and args.table_path and args.dataset_version:
    #     model_handler.classify_data(args.img_path, args.table_path, args.dataset_version)

    # Initialize the ModelHandler with the provided arguments
    model_handler = ModelHandler(data_path=args.data_path, model_path=args.model_path)

    # Run the infer method using the image path as both the input and output directory
    model_handler.infer(img_path=args.img_path)

# python generate_labels.py --data_path datasets/combined_dataset/data.yaml --img_path datasets\ISIC2019\ISIC_2019_Training_Input --table_path datasets\ISIC2019\ISIC_2019_Training_GroundTruth.csv --dataset_version ISIC2019
# python generate_labels.py --data_path datasets/combined_dataset/data.yaml --img_path datasets\ISIC2020\train --table_path datasets\ISIC2020\ISIC2020_transformed.csv --dataset_version ISIC2020
# python generate_labels.py --data_path datasets\combined_dataset\data.yaml --model_path runs\segment\train\weights\best.pt --img_path datasets\combined_dataset\2019_ISIC