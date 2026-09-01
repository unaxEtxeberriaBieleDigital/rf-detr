import uvicorn
from visualizer.backend.app import app

if __name__ == "__main__":
    # Inicia el servidor en el puerto 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)