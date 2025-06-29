# Live Face Recognition

A real-time facial recognition application built with Python, Gradio, and InsightFace. This application allows you to register faces, perform live recognition, and manage registered faces through an intuitive web interface.

Want to see it in action?
[Try it live on HuggingFace Spaces](https://huggingface.co/spaces/br1rda/live-face-recognition)


## Features

- **Real-time Face Recognition**: Live camera feed with instant face detection and recognition
- **Face Registration**: Register new faces with names for future recognition
- **Face Management**: Rename or delete registered faces with password protection
- **High Accuracy**: Uses InsightFace's state-of-the-art face recognition models
- **MongoDB Storage**: Persistent storage of face embeddings with vector search capabilities
- **Modern UI**: Clean Gradio interface with tabbed navigation

## Prerequisites

- Python 3.11
- MongoDB instance (local or cloud)
- Webcam for face capture

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/bruno-rda/live-face-recognition.git
   cd live-face-recognition
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## MongoDB Setup

1. **Set up your database**
   - Create a MongoDB database (local or cloud instance)
   - Create a collection for storing face embeddings (you can name it whatever you prefer)

2. **Create vector search index**
   You need to create a vector search index on your embedding field to enable similarity searches. The goal is to allow MongoDB to efficiently find faces with similar embeddings using cosine similarity.
   
   In your MongoDB instance (through MongoDB Compass, Atlas UI, or shell), create a vector search index with these parameters:
   - **Field**: Your embedding field name
   - **Vector Size**: 512
   - **Metric**: cosine
   - **Index Name**: embedding_index (or any name you prefer)

3. **Configure environment variables**
   After setting up your database and index, update your `.env` file with the corresponding values:
   ```env
   # Mongo secrets
   MONGO_URI=mongodb://your_connection_string
   DATABASE_NAME=your_database_name
   COLLECTION_NAME=your_collection_name
   VECTOR_SEARCH_FIELD_PATH=your_embedding_field

   # App secrets
   ADMIN_PASSWORD=your_admin_password
   ```

## Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Open your browser**
   Navigate to the URL shown in the terminal (typically `http://localhost:7860`)

3. **Use the application**
   - **Predict Faces**: View live camera feed with face recognition
   - **Register New Face**: Add new faces to the system
   - **Manage Faces**: Rename or delete registered faces

## Configuration

The application can be configured through the `config.py` file:

- `INSIGHTFACE_MODEL_NAME`: Face recognition model ('buffalo_l' or 'buffalo_s')
- `INSIGHTFACE_PROVIDERS`: Execution providers (CPU or CUDA)
- `FACE_SEARCH_THRESHOLD`: Similarity threshold for face matching (0.75)
- `STREAM_INTERVAL`: Camera stream update interval (0.1 seconds)

## Project Structure

```
live-face-recognition/
├── app.py                 # Main Gradio application
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── db/
│   ├── database.py       # MongoDB connection
│   └── repositories.py   # Face data operations
└── services/
    ├── face_analyzer.py  # InsightFace integration
    └── face_service.py   # Business logic
```

## Technical Details

### Face Recognition Pipeline

1. **Face Detection**: Uses InsightFace's detection module to locate faces in images
2. **Feature Extraction**: Generates 512-dimensional embeddings for each detected face
3. **Similarity Matching**: Compares embeddings using cosine similarity
4. **Threshold Filtering**: Only accepts matches above the configured threshold

### Database Schema

The MongoDB collection stores face embeddings with the following structure:
```json
{
  "name": "string",
  "{{VECTOR_SEARCH_FIELD_PATH}}": [512-dimensional vector]
}
```

### Security Features

- Password protection for face management operations
- Duplicate face detection during registration
- Name uniqueness validation

## Performance Optimization

- Use GPU acceleration by setting `INSIGHTFACE_PROVIDERS` to `['CUDAExecutionProvider']`
- Adjust `STREAM_INTERVAL` for better performance vs. responsiveness
- Use 'buffalo_s' model for faster processing (less accurate than 'buffalo_l')