## embeddings_utils.py

from typing import List
from fastembed import TextEmbedding, ImageEmbedding
import os

TEXT_MODEL_NAME = "Qdrant/clip-ViT-B-32-text"
IMAGE_MODEL_NAME = "Qdrant/clip-ViT-B-32-vision"


def convert_text_to_embeddings(documents: List[str], embedding_model: str = TEXT_MODEL_NAME) -> List:
    print(f"Converting {len(documents)} text documents to embeddings")
    text_embedding_model = TextEmbedding(model_name=embedding_model)
    text_embeddings = list(text_embedding_model.embed(documents))  # Returns a generator of embeddings
    print(f"Text embedding conversion complete. Shape: {len(text_embeddings)}")
    return text_embeddings


def convert_image_to_embeddings(images: List[str], embedding_model: str = IMAGE_MODEL_NAME) -> List:
    print(f"Converting {len(images)} images to embeddings")
    # Check if all image paths exist
    for img_path in images:
        if not os.path.exists(img_path):
            print(f"WARNING: Image path does not exist: {img_path}")

    image_model = ImageEmbedding(model_name=embedding_model)
    try:
        images_embedded = list(image_model.embed(images))
        print(f"Image embedding conversion complete. Shape: {len(images_embedded)}")
        return images_embedded
    except Exception as e:
        print(f"Error during image embedding: {str(e)}")
        raise


# Search for similar text and get corresponding images as well
def search_similar_text(collection_name, client, query, limit=3):
    print(f"Searching for text similar to: '{query}'")
    text_model = TextEmbedding(model_name=TEXT_MODEL_NAME)
    search_query = text_model.embed([query])
    search_results = client.search(
        collection_name=collection_name,
        query_vector=('text', list(search_query)[0]),
        with_payload=['image_path', 'caption'],
        limit=limit,
    )
    print(f"Found {len(search_results)} text matches")
    return search_results


# Search for similar images and get corresponding text as well
def search_similar_image(collection_name, client, query_image_path, limit=3):
    print(f"Searching for images similar to: {query_image_path}")
    if not os.path.exists(query_image_path):
        print(f"ERROR: Query image path does not exist: {query_image_path}")
        return []

    # Convert the query image into an embedding using the same model used for image embeddings
    image_embedding_model = ImageEmbedding(model_name=IMAGE_MODEL_NAME)

    try:
        # Embed the provided query image (assumed to be a file path)
        print(f"Generating embedding for query image")
        query_image_embedding = list(image_embedding_model.embed([query_image_path]))[
            0]  # Embedding for the query image

        # Perform the similarity search in the Qdrant collection for image embeddings
        search_results = client.search(
            collection_name=collection_name,
            query_vector=('image', query_image_embedding),
            with_payload=['image_path', 'caption'],  # Fetch image paths and captions as metadata
            limit=limit,
        )
        print(f"Found {len(search_results)} image matches")
        return search_results
    except Exception as e:
        print(f"Error during image search: {str(e)}")
        return []


def merge_results(text_results, image_results):
    # Combine based on some metadata, or simply concatenate
    combined_results = text_results + image_results
    print(f"Merged results: {len(text_results)} text + {len(image_results)} image = {len(combined_results)} total")
    return combined_results