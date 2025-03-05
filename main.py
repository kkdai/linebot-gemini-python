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
import google.generativeai as genai
import os
import sys
from io import BytesIO
import aiohttp
import PIL.Image

# Import LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate


# get channel_secret and channel_access_token from your environment variable
channel_secret = os.getenv('ChannelSecret', None)
channel_access_token = os.getenv('ChannelAccessToken', None)
gemini_key = os.getenv('GEMINI_API_KEY')
imgage_prompt = '''
Describe this image with scientific detail, reply in zh-TW:
'''

if channel_secret is None:
    print('Specify ChannelSecret as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify ChannelAccessToken as environment variable.')
    sys.exit(1)
if gemini_key is None:
    print('Specify GEMINI_API_KEY as environment variable.')
    sys.exit(1)

# Initialize the FastAPI app for LINEBot
app = FastAPI()
session = aiohttp.ClientSession()
async_http_client = AiohttpAsyncHttpClient(session)
line_bot_api = AsyncLineBotApi(channel_access_token, async_http_client)
parser = WebhookParser(channel_secret)

# Initialize LangChain with Gemini
os.environ["GOOGLE_API_KEY"] = gemini_key

# Create LangChain Gemini model instances
text_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
vision_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")


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
            # Process text message using LangChain
            msg = event.message.text
            response = generate_text_with_langchain(f'{msg}, reply in zh-TW:')
            reply_msg = TextSendMessage(text=response)
            await line_bot_api.reply_message(
                event.reply_token,
                reply_msg
            )
        elif (event.message.type == "image"):
            message_content = await line_bot_api.get_message_content(
                event.message.id)
            image_content = b''
            async for s in message_content.iter_content():
                image_content += s
            img = PIL.Image.open(BytesIO(image_content))

            result = generate_vision_with_langchain(img, imgage_prompt)
            reply_msg = TextSendMessage(text=result)
            await line_bot_api.reply_message(
                event.reply_token,
                reply_msg
            )
            return 'OK'
        else:
            continue

    return 'OK'


def generate_text_with_langchain(prompt):
    """
    Generate a text completion using LangChain with Gemini model.
    """
    # Create a chat prompt template with system instructions
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="You are a helpful assistant that responds in Traditional Chinese (zh-TW)."),
        HumanMessage(content=prompt)
    ])

    # Format the prompt and call the model
    formatted_prompt = prompt_template.format_messages()
    response = text_model.invoke(formatted_prompt)

    return response.content


def generate_vision_with_langchain(img, prompt):
    """
    Generate a image vision result using LangChain with Gemini model.
    """
    # Create a message with both text and image
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image", "image": img}
        ]
    )

    # Call the vision model
    response = vision_model.invoke([message])

    return response.content
