# SkinGardenia

A web application for analyzing skin conditions using AI and providing herbal recommendations.

## About

SkinGardenia is an AI-powered skin support app that combines image-based skin condition analysis with herbal care guidance. It includes a React frontend, a Node.js backend API, and a Python ML service for model inference. The goal is to provide quick, accessible, and informative skincare support while encouraging professional consultation when needed.

## Features

- AI-powered skin condition analysis
- Personalized herbal recommendations
- Modern, responsive UI

## Architecture

The application consists of three main components:

1. **Frontend**: React application with Vite
2. **Backend**: Node.js Express server
3. **ML Service**: Python service for running the TensorFlow model

## Setup and Installation

### Prerequisites

- Node.js (v16+)
- Python (v3.8+)
- npm or yarn

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/skin-herb-sanctuary-ai.git
   cd skin-herb-sanctuary-ai
   ```

2. Install frontend dependencies:
   ```
   npm install
   ```

3. Install backend dependencies:
   ```
   cd backend
   npm install
   ```

4. Install Python dependencies:
   ```
   cd backend
   pip install -r requirements.txt
   ```

## Running the Application

### Option 1: Using the startup scripts

#### macOS / Linux

Use the shell script from the project root:
```
chmod +x start-services.sh
./start-services.sh
```

#### Windows

Use the existing batch script:
```
start-services.bat
```

### Option 2: Manual startup

1. Start the ML Service:
   ```
   cd backend
   ./venv/bin/python3 src/ml_service.py
   ```

2. Start the Node Backend:
   ```
   cd backend
   npm run start
   ```

3. Start the Frontend:
   ```
   npm run dev
   ```

## Usage

1. Open your browser and navigate to `http://localhost:8080`
2. Go to the AI Skin Analyzer page
3. Upload an image of your skin
4. Click "Analyze Skin"
5. Wait for the analysis to complete
6. View your personalized recommendations

## API Endpoints

### Backend API

- `POST /api/analyze-skin`: Submit an image for analysis
- `GET /api/analysis-result/:requestId`: Get the result of an analysis
- `GET /health`: Health check endpoint

### ML Service API

- `POST /process`: Submit an image for processing
- `GET /result/:requestId`: Get the result of a processing request
- `GET /health`: Health check endpoint

## Troubleshooting

If you encounter any issues:

1. Make sure all services are running
2. Check the console logs for each service
3. Ensure the ML model file is in the correct location (`backend/Model/skin_conditions_model.h5`)
4. Check that all required Python packages are installed

## License

[MIT License](LICENSE)
