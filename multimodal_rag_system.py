## multimodal_rag_system.py

from src.create_data_embeddings import create_embeddings
from src.embeddings_utils import search_similar_text, search_similar_image, merge_results
from src.groq_utils import GroqClient  # New import for Groq client
import os

COLLECTION_NAME = "medical_images_text"


class MultimodalRAGSystem:
    collection_name: str

    def __init__(self):
        print("Initializing MultimodalRAGSystem...")
        self.groq_client = GroqClient()  # Changed from GPTClient to GroqClient
        try:
            self.qdrant_client = create_embeddings(COLLECTION_NAME)
            print(f"Collection {COLLECTION_NAME} created/loaded successfully")
            # Check if collection has points
            point_count = self.qdrant_client.count(COLLECTION_NAME).count
            print(f"Collection has {point_count} points")
        except Exception as e:
            print(f"ERROR initializing Qdrant collection: {str(e)}")
            raise
        self.collection_name = COLLECTION_NAME

    def process_query(self, query, query_image_path=None, top_k=3):
        print(f"\n--- Processing query: '{query}' ---")
        print(f"Query image path: {query_image_path}")
        if query_image_path and not os.path.exists(query_image_path):
            print(f"WARNING: Query image path does not exist: {query_image_path}")

        # 1. Text-based search if query is textual
        try:
            search_results_text = search_similar_text(self.collection_name, self.qdrant_client, query, limit=top_k)
            print(f"Text search found {len(search_results_text)} results")
        except Exception as e:
            print(f"ERROR in text search: {str(e)}")
            search_results_text = []

        # 2. Image-based search if a query image is provided
        search_results_image = []
        if query_image_path:  # Only perform image retrieval if an image path is provided
            try:
                search_results_image = search_similar_image(self.collection_name, self.qdrant_client, query_image_path,
                                                            limit=top_k)
                print(f"Image search found {len(search_results_image)} results")
            except Exception as e:
                print(f"ERROR in image search: {str(e)}")
                search_results_image = []

        # 3. Combine both results - merging text and image results
        combined_results = merge_results(search_results_text, search_results_image)
        print(f"Total combined results: {len(combined_results)}")

        # 4. Query Groq with the context and images
        try:
            print("Calling Groq API...")
            groq_response = self.groq_client.query(query, combined_results, query_image_path)
            print("Groq API call completed")
        except Exception as e:
            print(f"ERROR in Groq API call: {str(e)}")
            return f"Error: Could not get a response from the medical assistant. Details: {str(e)}"

        # 5. Process and return the response
        try:
            response = self.groq_client.process_response(groq_response)
            print(f"Response processed, length: {len(response) if response else 'None'}")
            return response
        except Exception as e:
            print(f"ERROR processing response: {str(e)}")
            return f"Error: Could not process the response. Details: {str(e)}"