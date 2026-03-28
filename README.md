# FurnishFrame

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

The server proxies generation requests to Google's Gemini image API using `gemini-2.5-flash-image` by default.

The API key should stay in an environment variable, not in frontend files.

## Deploy to Vercel

The frontend is static and the API now runs through Vercel Python Functions:

- `api/analyze.py`
- `api/generate.py`
- `vercel.json`

Deploy steps:

```bash
npm i -g vercel
vercel
vercel env add GEMINI_API_KEY
vercel --prod
```

Optional environment variables:

- `FURNISHFRAME_GEMINI_MODEL`
- `FURNISHFRAME_ANALYSIS_MODEL`

After the first `vercel` link step, subsequent production deploys are just:

```bash
vercel --prod
```
