## main.py

import gradio as gr
from PIL import Image
import numpy as np
import os
import sys
import logging
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("multimodal_assistant")

# Import after logging is set up
from src.multimodal_rag_system import MultimodalRAGSystem
from src.groq_utils import GroqClient

# Create a temporary directory for uploads if it doesn't exist
TEMP_DIR = Config.TEMP_DIR
os.makedirs(TEMP_DIR, exist_ok=True)
logger.info(f"Temporary directory: {TEMP_DIR}")

# Initialize the MultimodalRAGSystem
logger.info("Initializing MultimodalRAGSystem...")
try:
    system = MultimodalRAGSystem()
    logger.info("MultimodalRAGSystem initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MultimodalRAGSystem: {str(e)}")
    system = None


def test_groq_api():
    """Test function for Groq API connection"""
    logger.info("Testing Groq API connection...")
    try:
        client = GroqClient()
        response = client.query("Test message to verify API connection.", [], None)
        result = client.process_response(response)
        logger.info("Groq API test completed successfully")
        return result
    except Exception as e:
        logger.error(f"Groq API test failed: {str(e)}")
        return f"Error: {str(e)}"


# Define the Gradio function that will process the user input and image
def chatbot_interface(user_query, user_image=None):
    """Process user query and optional image input"""
    logger.info(f"Processing user query: {user_query}")
    logger.info(f"User provided image: {user_image is not None}")

    try:
        if user_image is not None:
            # If image is a numpy array (from Gradio), convert to PIL Image
            if isinstance(user_image, np.ndarray):
                logger.info(f"Converting numpy array image of shape {user_image.shape}")
                user_image = Image.fromarray(user_image)

            # Save the image temporarily
            user_image_path = os.path.join(TEMP_DIR, "temp_upload.jpg")
            user_image.save(user_image_path)
            logger.info(f"Saved user image to: {user_image_path}")
        else:
            user_image_path = None
            logger.info("No user image provided")

        # Check if system is initialized
        if system is None:
            logger.error("MultimodalRAGSystem not initialized")
            return "Error: System not initialized properly. Check logs for details."

        # Get the response from the Multimodal AI system
        logger.info("Calling process_query on MultimodalRAGSystem")
        response = system.process_query(user_query, query_image_path=user_image_path)
        logger.info("Received response from system")
        return response
    except Exception as e:
        logger.error(f"Error in chatbot_interface: {str(e)}")
        return f"Error processing your request: {str(e)}"


# Create the Gradio interface with text input and image input
# Create the Gradio interface with text input, image input, and a test button
interface = gr.Interface(
    fn=chatbot_interface,
    inputs=[
        gr.components.Textbox(lines=5, label="User Query", placeholder="Ask a medical question..."),
        gr.components.Image(label="Upload Medical Image", type="pil")
    ],
    outputs=gr.components.Textbox(label="AI Response"),
    title="Multimodal Medical Assistant",
    description="Ask medical-related questions and upload relevant medical images for analysis.",
    examples=[
        ["Can you describe what you see in this X-ray?", None],
        ["What might be causing the abnormality in this scan?", None],
        ["Is there any fracture visible in this image?", None]
    ]
)

# Add a separate interface for testing the API connection
test_interface = gr.Interface(
    fn=test_groq_api,
    inputs=[],
    outputs=gr.components.Textbox(label="API Test Result"),
    title="Test Groq API Connection",
    description="Click submit to test the connection to the Groq API."
)


# Add a utility function to check collection status
def check_collection_status():
    """Check the status of the Qdrant collection"""
    try:
        from src.create_data_embeddings import create_embeddings
        from src.multimodal_rag_system import COLLECTION_NAME

        client = create_embeddings(COLLECTION_NAME)
        collection_info = client.get_collection(COLLECTION_NAME)
        point_count = client.count(COLLECTION_NAME).count

        return f"Collection info: {collection_info}\nPoints in collection: {point_count}"
    except Exception as e:
        return f"Error checking collection status: {str(e)}"


# Add a diagnostic interface
diagnostic_interface = gr.Interface(
    fn=check_collection_status,
    inputs=[],
    outputs=gr.components.Textbox(label="Collection Status"),
    title="Check Collection Status",
    description="Check the status of the vector database collection."
)

# Launch the interfaces
if __name__ == "__main__":
    logger.info("Starting Gradio application...")
    gr.TabbedInterface(
        [interface, test_interface, diagnostic_interface],
        ["Medical Assistant", "API Test", "Diagnostics"]
    ).launch(debug=True)
    logger.info("Gradio application stopped")