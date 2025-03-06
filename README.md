# Gemini Helper with LangChain and Vertex AI

## Project Background

This project is a LINE bot that uses Google's Vertex AI Gemini models through LangChain to generate responses to both text and image inputs. The bot can answer questions in Traditional Chinese and provide detailed descriptions of images.

## Screenshot

![image](https://github.com/kkdai/linebot-gemini-python/assets/2252691/466fbe7c-e704-45f9-8584-91cfa2c99e48)

## Features

- Text message processing using Gemini AI in Traditional Chinese
- Image analysis with scientific detail in Traditional Chinese
- Integration with LINE Messaging API for easy mobile access
- Built with FastAPI for efficient asynchronous processing

## Technologies Used

- Python 3
- FastAPI
- LINE Messaging API
- Google Vertex AI (Gemini 2.0 Flash)
- LangChain
- Aiohttp
- PIL (Python Imaging Library)

## Setup

1. Clone the repository to your local machine.
2. Set up Google Cloud:
   - Create a Google Cloud project
   - Enable Vertex AI API
   - Set up authentication (service account or application default credentials)

3. Set the following environment variables:
   - `ChannelSecret`: Your LINE channel secret
   - `ChannelAccessToken`: Your LINE channel access token
   - `GOOGLE_PROJECT_ID`: Your Google Cloud Project ID
   - `GOOGLE_LOCATION`: Google Cloud region (default: us-central1)
   - Optional: `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account key file (if running locally)

4. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

5. Start the FastAPI server:

   ```
   uvicorn main:app --reload
   ```

6. Set up your LINE bot webhook URL to point to your server's endpoint.

## Usage

### Text Processing

Send any text message to the LINE bot, and it will use Vertex AI's Gemini model to generate a response in Traditional Chinese.

### Image Processing

Send an image to the bot, and it will analyze and describe the image with scientific detail in Traditional Chinese.

## Deployment Options

### Local Development

Use ngrok or similar tools to expose your local server to the internet for webhook access:

### Google Cloud Run

1. Install the Google Cloud SDK and authenticate with your Google Cloud account.
2. Build the Docker image:

   ```
   gcloud builds submit --tag gcr.io/$GOOGLE_PROJECT_ID/linebot-gemini
   ```

3. Deploy the Docker image to Cloud Run:

   ```
   gcloud run deploy linebot-gemini --image gcr.io/$GOOGLE_PROJECT_ID/linebot-gemini --platform managed --region $GOOGLE_LOCATION --allow-unauthenticated
   ```

4. Set up your LINE bot webhook URL to point to the Cloud Run service URL.
