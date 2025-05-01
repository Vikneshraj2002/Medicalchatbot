## create_data_embeddings.py

import os
import uuid
import pandas as pd
import random
from fastembed import TextEmbedding, ImageEmbedding
from qdrant_client import QdrantClient, models
from src.embeddings_utils import convert_text_to_embeddings, convert_image_to_embeddings, TEXT_MODEL_NAME, \
    IMAGE_MODEL_NAME
from config import Config

# Get base data path from config
DATA_PATH = Config.DATA_PATH

# Set sampling rate
SAMPLE_RATE = 0.5  # Use 10% of the data


def create_uuid_from_image_id(image_id):
    NAMESPACE_UUID = uuid.UUID('12345678-1234-5678-1234-567812345678')
    return str(uuid.uuid5(NAMESPACE_UUID, image_id))


def create_embeddings(collection_name):
    print(f"Creating/loading embeddings for collection: {collection_name}")
    print(f"Using data path: {DATA_PATH}")

    # Check if data path exists
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data path does not exist: {DATA_PATH}")
        return QdrantClient(":memory:")  # Return empty client

    # Initialize list to store all image documents from different splits
    all_image_docs = []

    # Process each data split: test, train, validation
    for split in ['test', 'train', 'valid']:
        # Define paths for images and captions
        images_folder = 'test_images' if split == 'test' else ('train_images' if split == 'train' else 'valid_images')
        images_path = os.path.join(DATA_PATH, images_folder, split)
        caption_file = os.path.join(DATA_PATH, f"{split}_captions.csv")  # Assuming naming convention

        # Skip if path doesn't exist
        if not os.path.exists(caption_file):
            print(f"Skipping {split} - caption file not found: {caption_file}")
            continue

        if not os.path.exists(images_path):
            print(f"Skipping {split} - images directory not found: {images_path}")
            continue

        # Read captions
        try:
            # First try with column names 'ID' and 'Caption' as mentioned in your feedback
            caption_df = pd.read_csv(caption_file)
            print(f"Read caption file: {caption_file}, shape: {caption_df.shape}")
            # Check if we have the expected columns and rename if needed
            if 'ID' in caption_df.columns and 'Caption' in caption_df.columns:
                # Rename to match our expected column names
                caption_df = caption_df.rename(columns={'ID': 'image_id', 'Caption': 'caption'})
                print("Renamed columns 'ID' to 'image_id' and 'Caption' to 'caption'")
            elif 'image_id' not in caption_df.columns or 'caption' not in caption_df.columns:
                print(f"CSV file {caption_file} does not have expected columns. Adjusting...")
                # Try to infer column names based on the first few rows
                caption_df = pd.read_csv(caption_file, header=None)
                if len(caption_df.columns) >= 2:
                    caption_df.columns = ['image_id', 'caption'] + [f'col_{i}' for i in
                                                                    range(2, len(caption_df.columns))]
                    print(f"Inferred column names from header-less CSV: {caption_df.columns}")
                else:
                    print(f"Cannot process {caption_file} - not enough columns")
                    continue
        except Exception as e:
            print(f"Error reading captions from {caption_file}: {e}")
            continue

        # Get list of available images
        try:
            image_files = os.listdir(images_path)
            print(f"Found {len(image_files)} images in {images_path}")
            # Randomly sample images
            sampled_image_files = random.sample(image_files, max(1, int(len(image_files) * SAMPLE_RATE)))
            print(f"Sampled {len(sampled_image_files)} out of {len(image_files)} images for {split}")
        except Exception as e:
            print(f"Error listing or sampling images in {images_path}: {e}")
            continue

        # Match images with captions
        image_docs_for_split = []
        for image_file in sampled_image_files:
            # Extract image_id from filename (assuming format like "image_id.jpg")
            image_id = image_file.split('.')[0]

            # Check if this image_id has a caption
            if image_id in caption_df['image_id'].values:
                caption = caption_df[caption_df['image_id'] == image_id]['caption'].values[0]
                image_path = os.path.join(images_path, image_file)

                # Check if image file exists
                if not os.path.exists(image_path):
                    print(f"WARNING: Image file does not exist: {image_path}")
                    continue

                # Add to our documents
                image_docs_for_split.append({
                    'image_id': image_id,
                    'caption': caption,
                    'image_path': image_path,
                    'split': split
                })

        print(f"Found {len(image_docs_for_split)} matching sampled images in {split}")
        all_image_docs.extend(image_docs_for_split)

    print(f"Total sampled images found across all splits: {len(all_image_docs)}")

    # If no images found, return empty client
    if len(all_image_docs) == 0:
        print("WARNING: No images found. Check your data paths and file structure.")
        return QdrantClient(":memory:")

    # Process in batches to avoid memory issues
    batch_size = 50  # Reduce batch size to avoid memory issues

    # Initialize client
    client = QdrantClient(":memory:")
    print("Initialized Qdrant client in memory")

    # Define vector dimensions for CLIP models (we know these dimensions)
    # CLIP ViT-B-32 has 512-dimensional embeddings for both text and images
    text_embeddings_size = 512
    image_embeddings_size = 512

    # Create collection if it doesn't exist
    if not client.collection_exists(collection_name):
        print(f"Creating new collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "image": models.VectorParams(size=image_embeddings_size, distance=models.Distance.COSINE),
                "text": models.VectorParams(size=text_embeddings_size, distance=models.Distance.COSINE),
            }
        )
    else:
        print(f"Collection {collection_name} already exists")

    # Process in batches
    for i in range(0, len(all_image_docs), batch_size):
        batch = all_image_docs[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} of {(len(all_image_docs) + batch_size - 1) // batch_size}")

        # Extract captions and image paths
        captions = [doc['caption'] for doc in batch]
        image_paths = [doc['image_path'] for doc in batch]

        try:
            # Generate embeddings
            caption_embeddings = convert_text_to_embeddings(captions)
            image_embeddings = convert_image_to_embeddings(image_paths)

            # Create points for upload
            points = []
            for j, doc in enumerate(batch):
                points.append(
                    models.PointStruct(
                        id=create_uuid_from_image_id(doc['image_id']),
                        vector={
                            "text": caption_embeddings[j],
                            "image": image_embeddings[j],
                        },
                        payload={
                            "image_id": doc['image_id'],
                            "caption": doc['caption'],
                            "image_path": doc['image_path'],
                            "split": doc['split']
                        }
                    )
                )

            # Upload batch
            client.upload_points(collection_name=collection_name, points=points)
            print(f"Successfully uploaded batch {i // batch_size + 1}")
        except Exception as e:
            print(f"ERROR processing batch {i // batch_size + 1}: {str(e)}")
            continue

    # Check final collection size
    count = client.count(collection_name).count
    print(f"Final collection size: {count} points")
    return client