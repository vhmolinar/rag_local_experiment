#!/bin/bash
# start_ollama.sh
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS=.ollama_models

# Create models directory if it doesn't exist
mkdir -p .ollama_models

if [ -f ollama.pid ]; then
    PID=$(cat ollama.pid)
    if ps -p $PID > /dev/null; then
        echo "Ollama is already running with PID $PID"
        exit 0
    else
        rm ollama.pid
    fi
fi

echo "Starting Ollama server..."
nohup ./bin/ollama serve > ollama.log 2>&1 &
disown
echo $! > ollama.pid

# Wait for Ollama to become responsive
echo "Waiting for Ollama to start..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:11434/api/tags > /dev/null; then
        echo "Ollama is ready!"
        exit 0
    fi
    sleep 1
done

echo "Ollama did not start in time. Check ollama.log for details."
exit 1
