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

The server proxies generation requests to Google's Gemini image API using `gemini-3.1-flash-image-preview` by default.

#Inspiration

FurnishFrame was inspired by the challenge many people face when trying to visualize how a space could look before committing to furniture or design changes. Traditional interior design tools can be time-consuming or require expertise, so we wanted to create an intuitive, AI-powered solution that makes design exploration accessible, fast, and engaging for anyone. Many inspirations came from home-building game aspects.

#What the application does

FurnishFrame is an AR-powered room staging tool that transforms uploaded photos into fully reimagined interior spaces. Users can experiment with different layouts, furniture styles, and décor preferences, and the app generates realistic, styled environments tailored to their vision. It allows users to quickly visualize multiple design possibilities before making real-world decisions.

#How it was built

The application is powered by Nano Banana 2, which enables advanced image generation and transformation. We developed a prompting system that interprets user input and translates it into detailed instructions for staging rooms. The system combines contextual understanding of room structure with style-based generation to produce cohesive and realistic outputs.

#Challenges we ran into

One of the main challenges was refining the prompting system to accurately interpret user instructions. Translating abstract preferences into precise object placement while maintaining spatial realism required multiple iterations. Ensuring that furniture and décor aligned naturally with the room’s layout and with each other, was another key technical hurdle.

#How are project isn't like other applications

Unlike many interior design tools that rely on static templates or manual placement, FurnishFrame dynamically generates fully staged environments based on user input. Its ability to combine AR, AI-driven generation, and contextual understanding allows for more personalized and realistic results, making it stand out from traditional design apps.

#What we are proud of

We are proud of creating a seamless and intuitive user experience that bridges creativity and technology. The app empowers users to explore design ideas effortlessly and produces high-quality, realistic transformations that feel both practical and inspiring.

#What we learned

Through building FurnishFrame, we learned the importance of prompt engineering in AI-driven applications. Small changes in how instructions are structured can significantly impact output quality. We also gained insights into balancing user control with automated generation to achieve both flexibility and coherence.

