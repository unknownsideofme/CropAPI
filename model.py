import os
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch

class ImageCaptionTool(BaseTool):
    name: str = "Image captioner"
    description: str = (
        "Use this tool when given the path to an image that you would like to be described."
    )

    def _run(self, img_path: str) -> str:
        image = Image.open(img_path).convert("RGB")

        model_name = "Salesforce/blip-image-captioning-large"
        device = "cpu"  # Change to "cuda" if GPU is available

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        inputs = processor(image, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=20)

        caption = processor.decode(output[0], skip_special_tokens=True)

        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

class ObjectDetectionTool(BaseTool):
    name: str = "Object detector"
    description: str = (
        "Use this tool when given the path to an image that you would like to detect objects. "
        "It will return a list of all detected objects. Each element in the list is in the format: "
        "[x1, y1, x2, y2] class_name confidence_score."
    )

    def _run(self, img_path: str) -> str:
        image = Image.open(img_path).convert("RGB")

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Convert outputs (bounding boxes and class logits) to COCO API
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += "[{}, {}, {}, {}]".format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            detections += " {}".format(model.config.id2label[int(label)])
            detections += " {}\n".format(float(score))

        return detections

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

def parse_response_to_json(response_text: str) -> Dict[str, Any]:
    """
    Parse the LLM response and ensure it's in the required JSON format.
    If parsing fails, create a structured response with an error message.
    """
    # First, try to find JSON objects in the text (they might be embedded in other text)
    try:
        # Check if there's a JSON object in the text
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > 0:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
        
        # If we can't find JSON brackets, try to create the structure manually
        result = {
            "possible_diagnosis": [],
            "causes": [],
            "remedies_or_cure": []
        }
        
        # Try to extract information from the text response
        if "diagnosis" in response_text.lower():
            diagnosis_section = response_text.lower().split("diagnosis")[1].split("causes")[0]
            result["possible_diagnosis"] = [d.strip() for d in diagnosis_section.split("\n") if d.strip()]
            
        if "causes" in response_text.lower():
            causes_section = response_text.lower().split("causes")[1].split("remedies")[0]
            result["causes"] = [c.strip() for c in causes_section.split("\n") if c.strip()]
            
        if "remedies" in response_text.lower() or "cure" in response_text.lower():
            remedies_section = response_text.lower().split("remedies" if "remedies" in response_text.lower() else "cure")[1]
            result["remedies_or_cure"] = [r.strip() for r in remedies_section.split("\n") if r.strip()]
        
        # If we still have empty sections, include the raw response
        if not any(result.values()):
            result["raw_response"] = response_text
            
        return result
    
    except Exception as e:
        # If all parsing fails, return a structured error response
        return {
            "possible_diagnosis": ["Unable to determine diagnosis"],
            "causes": ["Unable to parse causes"],
            "remedies_or_cure": ["Unable to parse remedies"],
            "error": str(e),
            "raw_response": response_text
        }

def process_plant_disease_image(
    image_path: str, 
    disease_symptoms: Optional[str] = None,
    image_dir: str = "images/"
) -> Dict[str, Any]:
    """
    Process an image for plant disease detection and return a structured JSON response.
    
    Args:
        image_path (str): Name of the image file (will be joined with image_dir)
        disease_symptoms (str, optional): Description of disease symptoms
        image_dir (str, optional): Directory where images are stored. Defaults to "images/".
    
    Returns:
        Dict[str, Any]: A JSON-formatted dictionary with diagnosis, causes, and remedies
    """
    # Load environment variables and API key
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if api_key is None:
        raise ValueError("API Key not set. Please set the GROQ_API_KEY environment variable.")

    # Initialize LLM
    llm = ChatGroq(api_key=api_key, model="llama-3.2-90b-vision-preview")

    # Full path to the image
    full_image_path = os.path.join(image_dir, image_path)
    
    # Validate image exists
    if not os.path.exists(full_image_path):
        return {
            "error": f"Image not found at path: {full_image_path}",
            "possible_diagnosis": [],
            "causes": [],
            "remedies_or_cure": []
        }

    # Default symptoms if not provided
    if disease_symptoms is None:
        disease_symptoms = """Disease Symptoms
1. Wet-looking, dark patches appear on leaves, usually starting from the edges.
2. A white, cotton-like layer may grow on the underside of leaves when it's humid."""

    # Create a structured prompt with explicit JSON output format
    prompt = ChatPromptTemplate.from_template(
        """
        You are a plant disease diagnosis expert. Analyze the image described to you and the provided symptoms.

        Symptoms:
        {symptoms}
        
        Based on the image at '{image_path}' and the symptoms, provide your analysis as a valid JSON object with the following structure:
        {{
            "possible_diagnosis": ["Diagnosis 1", "Diagnosis 2"],
            "causes": ["Cause 1", "Cause 2"],
            "remedies_or_cure": ["Remedy 1", "Remedy 2"]
        }}
        
        You MUST respond with valid JSON and nothing else. No explanations, no markdown formatting.
        """
    )

    # Initialize tools
    tools = [ImageCaptionTool(), ObjectDetectionTool()]

    # Setup conversational memory
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=3,
        return_messages=True
    )

    # Initialize agent with our custom prompt
    agent = initialize_agent(
        agent="chat-conversational-react-description",
        tools=tools,
        llm=llm,
        max_iterations=5,
        verbose=False,
        memory=conversational_memory,
        early_stopping_method='generate'
    )

    try:
        # Process the image using the agent
        response = agent.invoke({
            "input": prompt.format(
                symptoms=disease_symptoms,
                image_path=full_image_path
            )
        })
        
        # Get the output text
        output_text = response["output"]
        
        # Parse and validate the response is in JSON format
        result = parse_response_to_json(output_text)
        
        # Add metadata
        result["image_path"] = image_path
        result["success"] = True
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "image_path": image_path,
            "possible_diagnosis": [],
            "causes": [],
            "remedies_or_cure": []
        }

# Function to use in FastAPI app
async def process_uploaded_image(file_path: str, symptoms: Optional[str] = None) -> Dict[str, Any]:
    """
    Process an uploaded image for disease detection. This function is designed to be used in a FastAPI app.
    
    Args:
        file_path (str): Path to the uploaded image file
        symptoms (str, optional): Description of disease symptoms
        
    Returns:
        Dict[str, Any]: JSON response with diagnosis information
    """
    try:
        # Use the processing function with the file path directly
        # Note: we're not using image_dir here since we have the full path
        result = process_plant_disease_image(
            image_path=file_path,
            disease_symptoms=symptoms,
            image_dir=""  # Empty string because file_path is already the full path
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "image_path": file_path,
            "possible_diagnosis": [],
            "causes": [],
            "remedies_or_cure": []
        }