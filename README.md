# Multimodal Medical Assistant

A sophisticated AI-driven medical imaging analysis system that combines text and image search capabilities to help medical professionals interpret medical images and provide diagnostic insights.

## 🌟 Features

- **Multimodal RAG (Retrieval-Augmented Generation)**: Combines text and image retrieval to find relevant medical cases
- **Advanced AI Analysis**: Uses Groq's LLaMa vision models to analyze medical images with expert context
- **Vector Database**: Stores embeddings for efficient similarity search of medical images and descriptions
- **User-Friendly Interface**: Simple Gradio web interface for text queries and image uploads
- **Domain-Specific Knowledge**: Specialized in medical imaging and radiological analysis

## 🔧 Technical Architecture

The system is built on several key components:

1. **Embedding Generation**: Uses FastEmbed with CLIP models for both text and image embedding
2. **Vector Storage**: Qdrant vector database for efficient similarity search
3. **LLM Integration**: Groq API with LLaMa vision models for multimodal analysis
4. **RAG System**: Custom retrieval system that combines text and image similarity search
5. **Web Interface**: Gradio-based UI for easy interaction

## 📋 Prerequisites

- Python 3.8+
- Groq API key
- Medical image dataset (such as ROCO v2)
- Required Python libraries (see `requirements.txt`)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multimodal-medical-assistant.git
cd multimodal-medical-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your API keys and paths in `config.py`:
```python
# Update with your actual API key and paths
GROQ_API_KEY = "your-groq-api-key-here"
DATA_PATH = '/path/to/your/medical/dataset/'
TEMP_DIR = '/path/to/temp/directory/'
```

## 📊 Data Setup
Link to RocoV2 dataset : https://github.com/sctg-development/ROCOv2-radiology.git

The system expects a dataset structured like the ROCO v2 dataset with:

- Image directories (`train_images`, `test_images`, `valid_images`)
- Caption files (`train_captions.csv`, `test_captions.csv`, `valid_captions.csv`)

Each CSV file should contain at least:
- `image_id`: Unique identifier for each image
- `caption`: Medical description of the image

## 📝 Usage

1. Start the application:
```bash
python main.py
```

2. Open your browser at the displayed URL (typically `http://127.0.0.1:7860`)

3. In the web interface:
   - Type your medical query in the text box
   - Optionally upload a medical image for analysis
   - Click "Submit" to receive the AI analysis

## 🧩 Key Components

### Embedding Utilities (`embeddings_utils.py`)
Handles text and image embedding generation using FastEmbed models.

### Groq Integration (`groq_utils.py`)
Manages communication with the Groq API for LLM processing.

### Vector Database Setup (`create_data_embeddings.py`)
Prepares and loads the medical dataset into the vector database.

### RAG System (`multimodal_rag_system.py`)
Core component that coordinates retrieval and generation processes.

## 🔍 Diagnostics

The application includes diagnostic tabs to:
- Test the Groq API connection
- Check the vector database collection status

## 📈 Performance Considerations

- The system samples a portion of the dataset (10% by default) to manage memory usage
- Processing is done in batches to avoid memory issues
- Consider increasing hardware resources for larger datasets

## 🛠️ Troubleshooting

- Check the `app.log` file for detailed error messages
- Ensure all paths in `config.py` are correctly set
- Verify your Groq API key is valid
- Confirm that your dataset structure matches the expected format

## 🔄 Future Improvements

- Fine-tuning the medical vision model for improved accuracy
- Adding support for additional medical imaging modalities
- Implementing user feedback mechanism for continuous improvement
- Adding authentication and user management features
- Support for DICOM and other specialized medical file formats

## 📄 License

[Specify your license here]

## 📞 Support

For issues, questions, or contributions, please contact [your contact information here].
