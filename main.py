from linebot.models import (
    MessageEvent, TextSendMessage
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.aiohttp_async_http_client import AiohttpAsyncHttpClient
from linebot import (
    AsyncLineBotApi, WebhookParser
)
from fastapi import Request, FastAPI, HTTPException
import os
import sys
from io import BytesIO
import aiohttp
import PIL.Image
import base64
import uuid
from google.cloud import storage


# Import LangChain components with Vertex AI
from langchain_google_vertexai import ChatVertexAI
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


# get channel_secret and channel_access_token from your environment variable
channel_secret = os.getenv('ChannelSecret', None)
channel_access_token = os.getenv('ChannelAccessToken', None)
imgage_prompt = '''
Describe this image with scientific detail, reply in zh-TW:
'''

# Vertex AI needs a project ID and possibly authentication
google_project_id = os.getenv('GOOGLE_PROJECT_ID')
# Location for Vertex AI resources, e.g., "us-central1"
google_location = os.getenv('GOOGLE_LOCATION', 'us-central1')
# Google Cloud Storage bucket for image uploads
google_storage_bucket = os.getenv('GOOGLE_STORAGE_BUCKET', None)


if channel_secret is None:
    print('Specify ChannelSecret as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify ChannelAccessToken as environment variable.')
    sys.exit(1)
if google_project_id is None:
    print('Specify GOOGLE_PROJECT_ID as environment variable.')
    sys.exit(1)
if google_storage_bucket is None:
    print('Specify GOOGLE_STORAGE_BUCKET as environment variable.')
    sys.exit(1)


# Initialize the FastAPI app for LINEBot
app = FastAPI()
session = aiohttp.ClientSession()
async_http_client = AiohttpAsyncHttpClient(session)
line_bot_api = AsyncLineBotApi(channel_access_token, async_http_client)
parser = WebhookParser(channel_secret)

# Using a single, powerful multimodal model for both text and images.
# gemini-2.0-flash is a powerful, cost-effective model for multimodal tasks.
model = ChatVertexAI(
    model_name="gemini-2.0-flash",
    project=google_project_id,
    location=google_location,
    max_output_tokens=2048  # Increased token limit for detailed image descriptions
)


def upload_to_gcs(file_stream, file_name, bucket_name):
    """Uploads a file to the bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        blob.upload_from_file(file_stream, content_type='image/jpeg')

        # Return the GCS URI
        return f"gs://{bucket_name}/{file_name}"
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        return None


def delete_from_gcs(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()
        print(f"Blob {blob_name} deleted from bucket {bucket_name}.")
    except Exception as e:
        print(f"Error deleting from GCS: {e}")


@app.post("/")
async def handle_callback(request: Request):
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = await request.body()
    body = body.decode()

    try:
        events = parser.parse(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if not isinstance(event, MessageEvent):
            continue

        if (event.message.type == "text"):
            # Process text message using LangChain with Vertex AI
            msg = event.message.text
            user_id = event.source.user_id
            print(f"Received message: {msg} from user: {user_id}")
            response = generate_text_with_langchain(f'{msg}, reply in zh-TW:')
            reply_msg = TextSendMessage(text=response)
            await line_bot_api.reply_message(
                event.reply_token,
                reply_msg
            )
        elif (event.message.type == "image"):
            user_id = event.source.user_id
            print(f"Received image from user: {user_id}")

            message_content = await line_bot_api.get_message_content(event.message.id)
            image_content = b''
            async for s in message_content.iter_content():
                image_content += s
            img_stream = PIL.Image.open(BytesIO(image_content))

            file_name = f"{uuid.uuid4()}.jpg"
            gcs_uri = None
            response = "抱歉，處理您的圖片時發生錯誤。"  # Default error message

            try:
                gcs_uri = upload_to_gcs(
                    img_stream, file_name, google_storage_bucket)
                if gcs_uri:
                    print(f"Image uploaded to {gcs_uri}")
                    response = generate_image_description(gcs_uri)
            finally:
                # Clean up the GCS file if it was uploaded
                if gcs_uri:
                    delete_from_gcs(google_storage_bucket, file_name)

            reply_msg = TextSendMessage(text=response)
            await line_bot_api.reply_message(
                event.reply_token,
                reply_msg
            )
        else:
            continue

    return 'OK'


def generate_text_with_langchain(prompt):
    """
    Generate a text completion using LangChain with Vertex AI model.
    """
    # Create a chat prompt template with system instructions
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="You are a helpful assistant that responds in Traditional Chinese (zh-TW)."),
        HumanMessage(content=prompt)
    ])

    # Format the prompt and call the model
    formatted_prompt = prompt_template.format_messages()
    response = model.invoke(formatted_prompt)

    return response.content


def generate_image_description(image_uri):
    """
    Generate a description for an image using LangChain with Vertex AI.
    """
    # The prompt is already defined globally as imgage_prompt
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": imgage_prompt
            },
            {
                "type": "image_url",
                "image_url": image_uri
            },
        ]
    )

    response = model.invoke([message])
    return response.content
