from groq import Groq
import base64
import os
from config import Config
import time


class GroqClient:
    def __init__(self):
        self.api_key = Config.GROQ_API_KEY
        self.client = Groq(api_key=self.api_key)
        print(f"GroqClient initialized with API key ending in: ...{self.api_key[-5:]}")

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def query(self, prompt, retrieved_contexts, user_image=None):
        print(f"GroqClient query: {prompt[:50]}...")
        print(f"Retrieved contexts: {len(retrieved_contexts)}")
        print(f"User image: {user_image}")

        # System role content (to be included as text in user message)
        radiologist_instructions = """You are a radiologist with an experience of 30 years.
        You analyse medical scans and text, and help diagnose underlying issues.

        Please analyze the following query and image using your expertise:
        """

        # Initialize message structure - use only user message (no system message)
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": f"{radiologist_instructions}\n\n{prompt}"}
            ]}
        ]

        # Add the user-uploaded image (if any)
        if user_image:
            if not os.path.exists(user_image):
                print(f"ERROR: User image file does not exist: {user_image}")
                return {"error": f"Image file not found: {user_image}"}

            try:
                base64_image = self.encode_image(user_image)
                print(f"Successfully encoded user image, size: {len(base64_image)}")
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            except Exception as e:
                print(f"ERROR encoding user image: {str(e)}")
                return {"error": f"Image encoding failed: {str(e)}"}

        # Add context message for retrieved images as text only
        if retrieved_contexts:
            context_text = "Additional context that you may use as a reference. Use them if you feel they are relevant to " \
                           "the case. NOTE: They are not the patient's images. They are descriptions of other patients' " \
                           "images which can be used as a reference, if required.\n\n"

            # Compile all context descriptions into a single text block
            for i, context in enumerate(retrieved_contexts):
                try:
                    caption = context.payload['caption']
                    image_path = context.payload['image_path']

                    # Add the caption and image reference (without sending the actual image)
                    context_text += f"Reference {i + 1}: {caption}\n"
                    # We can add image filename as additional reference
                    context_text += f"Source: {os.path.basename(image_path)}\n\n"

                except Exception as e:
                    print(f"ERROR processing context: {str(e)}")
                    continue

            # Add all contexts as a single text message
            messages[0]["content"].append({
                "type": "text",
                "text": context_text
            })

            print(f"Added {len(retrieved_contexts)} context items as text")

        # Call the API
        print("Sending request to Groq API...")
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=[m for m in messages],  # Convert to format expected by the client
                max_tokens=Config.MAX_TOKENS
            )
            elapsed_time = time.time() - start_time
            print(f"Groq API response received in {elapsed_time:.2f} seconds.")

            return {"choices": [{"message": {"content": response.choices[0].message.content}}]}
        except Exception as e:
            print(f"ERROR making API request: {str(e)}")
            return {"error": f"API request failed: {str(e)}"}

    def process_response(self, response):
        print(f"Processing response: {response.keys() if isinstance(response, dict) else 'Not a dict'}")

        if isinstance(response, dict) and 'error' in response:
            return f"Error: {response['error']}"

        if isinstance(response, dict) and 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            print(f"Successfully extracted content: {content[:50]}...")
            return content
        else:
            error_msg = f"Invalid response structure: {response}"
            print(error_msg)
            return f"Error: Unable to process response from API. Full response: {response}"