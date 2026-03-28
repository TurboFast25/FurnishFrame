# RoomVis

Basic framework for a room-staging web app that:

- uploads a room image
- lets users drag and drop furniture onto the scene
- supports selecting, moving, scaling, rotating, and deleting staged items
- calls Google Nano Banana through a local Python proxy so the API key stays out of browser code

## Files

- `index.html`: app shell
- `styles.css`: layout and visual styling
- `src/app.js`: room staging interactions
- `src/services/nanoBanana.js`: frontend API client
- `server.py`: static server + Gemini image generation proxy

## Run locally

This scaffold is dependency-free.

Example:

```bash
export GEMINI_API_KEY='your_key_here'
python3 server.py
```

Then visit `http://localhost:4173`.

The server proxies generation requests to Google's Gemini image API using `gemini-3.1-flash-image-preview`.

The API key should stay in an environment variable, not in frontend files.
