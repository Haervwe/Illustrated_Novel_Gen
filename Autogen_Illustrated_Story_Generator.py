import json
from typing import List, Optional,Dict
from PIL import Image as ImagePil
from typing import Optional
import asyncio
import os
from dataclasses import dataclass
from autogen_core.application import SingleThreadedAgentRuntime
from autogen_core.base import AgentId, MessageContext
from autogen_core.components import (
    RoutedAgent,
    TypeSubscription,
    message_handler,
)
from autogen_core.components.models import (
    ChatCompletionClient,
    LLMMessage,
    FunctionExecutionResult,
    SystemMessage,
    UserMessage,
)
from autogen_ext.models import OpenAIChatCompletionClient                                                                                                  
from autogen_core.components.tool_agent import ToolAgent, tool_agent_caller_loop, ToolException
from autogen_core.components.tools import FunctionTool, Tool, ToolSchema
from IPython.display import display  
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
import re
from datetime import datetime
import requests
import base64
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import Paragraph, SimpleDocTemplate ,Spacer, Image as ReportLabImage, Paragraph,Spacer, PageBreak
from io import BytesIO

#global Variables::

huggingface_api_key= "" # your api key leave it a blank if you have a environmet api key vairable setted
if not huggingface_api_key:
    huggingface_api_key = os.getenv('HF_API_KEY')
    if not huggingface_api_key :
        Console().print("No HF_API_KEY environment variable found, please set it to your api key in the above variable or in your environment keyring")



#sets de local file path to save function outputs
#set work_dir manually here.

work_dir = ""


# model Client Set up for local LLM:
# using custon class OllamaChatCompletion client wich  now it works with propper model flags and tool use in ollama, no more need of a hacky model name, and we can do multi model local inference!!

def get_model_client_tools() -> OpenAIChatCompletionClient:
    "Mimic OpenAI API using Local LLM Server."
    return OpenAIChatCompletionClient(
        model="llama3.1:latest",  
        api_key="NotRequiredSinceWeAreLocal",
        base_url="http://127.0.0.1:11434/v1",
        model_capabilities={
            "vision": False, # Replace with True if the model has vision capabilities.
            "function_calling": True, # Replace with True if the model has function calling capabilities.
            "json_output": True,  # Replace with True if the model has JSON output capabilities.
        },
        extra_create_args={
            "temperature": 0.1,  # Controls randomness (0.0 to 1.0)
            "top_p": 0.9,  # Limits token choices based on probability
            "top_k": 5,  # Limits token choices based on top K tokens
            "stream": False,  # Set to True if streaming is needed
            "frequency_penalty": 0.1,  # Penalizes frequent tokens (0.0 to 2.0)
            "presence_penalty": 0.1, # Penalizes presence of earlier tokens (0.0 to 2.0)
            "max_tokens": 128000
        },
       token_limit=128000
    )

def get_model_client_serious() -> OpenAIChatCompletionClient:
    "Mimic OpenAI API using Local LLM Server."
    return OpenAIChatCompletionClient(
        model= "hf.co/QuantFactory/Llama-3.2-3B-Overthinker-GGUF:Q8_0", 
        api_key="NotRequiredSinceWeAreLocal",
        base_url="http://127.0.0.1:11434/v1",
        model_capabilities={
            "vision": False, # Replace with True if the model has vision capabilities.
            "function_calling": False, # Replace with True if the model has function calling capabilities.
            "json_output": True,  # Replace with True if the model has JSON output capabilities.
        },
        token_limit=128000
    )
    
def get_model_client_creative() -> OpenAIChatCompletionClient:
    "Mimic OpenAI API using Local LLM Server."
    return OpenAIChatCompletionClient(
        model= "hf.co/mradermacher/CreativeSmart-2x7B-GGUF:Q4_K_S", 
        api_key="NotRequiredSinceWeAreLocal",
        base_url="http://127.0.0.1:11434/v1",
        model_capabilities={
            "vision": False, # Replace with True if the model has vision capabilities.
            "function_calling": False, # Replace with True if the model has function calling capabilities.
            "json_output": True,  # Replace with True if the model has JSON output capabilities.
        },
        extra_create_args={
            "temperature": 0.7,  # Controls randomness (0.0 to 1.0)
            "top_p": 0.9,  # Limits token choices based on probability
            "top_k": 10,  # Limits token choices based on top K tokens
            "stream": False,  # Set to True if streaming is needed
            "frequency_penalty": 0.3,  # Penalizes frequent tokens (0.0 to 2.0)
            "presence_penalty": 0.1, # Penalizes presence of earlier tokens (0.0 to 2.0)
            "max_tokens": 128000
        },
        token_limit=128000
    )

model_client_serious = get_model_client_serious()
model_client_tools = get_model_client_tools()
model_client_creative = get_model_client_creative()


#Message types definitions:
@dataclass
class Message:
    content: str  # Changed from Union[str, List[Union[str, Image]]]
    # Name of the agent that sent this message
    source: str

@dataclass
class ChapterMessage:
    content: Optional [str]  # Changed from Union[str, List[Union[str, Image]]]
    image: Optional [str]

@dataclass
class ToolAgentMessage:
    content: str

class Chapter(BaseModel):
    chapter_number: int
    guidelines: str

class NovelPlan(BaseModel):
    title: str
    chapters: List[Chapter]



#tool Definitions and utils

## utils for internal tool use

def create_folder(folder_name: str = None) -> str: #to be called inside a function tool, subsequent calls return the current project folder, takes an optional srt argument to defile the folder name, by default it uses system time..
    global work_dir
    if not work_dir:  # Check if work_dir is empty
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")

        if folder_name:
            folder_name = remove_unwanted_extensions(folder_name)
            folder_name = re.sub(r'[<>:"/\\|?*]', '_', folder_name)
        else: 
            folder_name = f"Output_{timestamp}"
        work_dir = os.path.join(os.getcwd(), 'output', folder_name)
        try:
            os.makedirs(work_dir)  # Create the directory if it doesn't exist
            #print(f"Created folder: {work_dir}")
        except OSError as e:
            print(f"Failed to create folder: {e}")


    return work_dir

def remove_unwanted_extensions(filename: str) -> str: #removes common file extentions potentially hallucianted by the LLM.
    # Define common image and text file extensions
    unwanted_extensions = [
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg',
        '.txt', '.csv', '.log', '.json', '.xml', '.md', '.doc', '.docx'
    ]

    # Get the base name without the directory path
    base_name = os.path.basename(filename)

    # Loop through unwanted extensions and remove them if present
    for ext in unwanted_extensions:
        if base_name.endswith(ext):
            # Remove the unwanted extension
            return base_name[:-len(ext)]  # Return the filename without the extension

    # If no unwanted extensions found, return the original filename
    return base_name

def save_image(image_bytes:str , file_name:str) -> str: # saves base64img and returns the same base64 encoded image or returns failure Execption.
    work_dir = create_folder()
    """
    Save the image bytes in the output directory.
    """
    cleansed_file_name = remove_unwanted_extensions(file_name)
    if not cleansed_file_name.strip():
        cleansed_file_name = datetime.now().strftime("image_%Y%m%d_%H%M%S")
    safe_filename = re.sub(r'[<>:"/\\|?*]', '_', cleansed_file_name) + ".jpg"
    save_path = os.path.join(work_dir, f"{safe_filename}")
    with open(save_path, 'wb') as image_file:
        image_file.write(image_bytes)
        if save_path and os.path.isfile(save_path):
                try:
                    image = ImagePil.open(save_path)
                    display(image)
                    #if succesfully generates the img with a HG api it pass the result as a base64 string to be compatible with the inner tool loop of the tool agent.
                    return base64.b64encode(image_bytes).decode('utf-8')
                except ValueError as e:
                    raise ToolException(call_id="image_display_error", content=f"Error displaying image: {e}")
        else:
            raise ToolException(call_id="file_not_found", content="Generated image file not found.")

def parse_json_output(llm_output: str) -> Optional[Dict]:
    """
    Parse JSON output from LLM response with improved error handling and validation.
    
    Args:
        llm_output (str): Raw string output from LLM
        
    Returns:
        Optional[Dict]: Parsed and validated JSON data or None if parsing fails
    """
    # Clean up the input string to handle common issues
    cleaned_output = llm_output.strip()
    
    # Try to find JSON content with improved regex
    json_pattern = r'({[\s\S]*})'
    json_match = re.search(json_pattern, cleaned_output)
    
    if not json_match:
        print("No JSON structure found in the output.")
        return None
        
    try:
        # Extract and parse JSON
        json_str = json_match.group(1)
        json_dict = json.loads(json_str)
        
        # Handle potential nested "format" structure
        if "format" in json_dict:
            json_dict = json_dict["format"]
            
        # Validate against our schema
        novel_plan = NovelPlan(**json_dict)
        return novel_plan.model_dump()
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None
    except Exception as e:
        print(f"Validation error: {e}")
        return None

def generate_chapter_prompts(json_data: Dict) -> List[str]:
    """
    Generate chapter prompts from parsed JSON data.
    
    Args:
        json_data (Dict): Validated JSON data
        
    Returns:
        List[str]: List of chapter prompts
    """
    chapter_prompts = []
    
    for chapter in json_data.get("chapters", []):
        prompt_text = (
            f"Chapter {chapter['chapter_number']}: {chapter['guidelines']}"
        )
        chapter_prompts.append(prompt_text)
    
    return chapter_prompts

# Functions to create the final PDF
def is_base64_encoded(data: Optional[str]) -> bool:
    if data is None:
        return False
    try:
        # Attempt to decode the base64 string
        decoded_data = base64.b64decode(data, validate=True)
        # Check if the decoded data is non-empty
        return len(decoded_data) > 0
    except (ValueError, TypeError):
        return False

def remove_chapter_mentions(content: str) -> str:
    """Remove mentions of 'Chapter ' or 'chapter' followed by a number from the first part of the content."""
    # Split the content into lines
    lines = content.splitlines()
    
    # Check if there's at least one line to process
    if lines:
        # Remove chapter mentions only from the first line
        lines[0] = re.sub(r'\bChapter\s+\d+\b', '', lines[0], flags=re.IGNORECASE).strip()
    
    # Join the lines back together
    return '\n'.join(lines)

def remove_chapter_mentions(content: str) -> str:
    """Remove mentions of 'Chapter ' or 'chapter' followed by a number from the first part of the content."""
    # Split the content into lines
    lines = content.splitlines()
    
    # Check if there's at least one line to process
    if lines:
        # Remove chapter mentions only from the first line
        lines[0] = re.sub(r'\bChapter\s+\d+\b', '', lines[0], flags=re.IGNORECASE).strip()
    
    # Join the lines back together
    return '\n'.join(lines)

def create_pdf(book: List[ChapterMessage], output_file: str):
    """Generate a PDF from a list of chapter messages, saving it in the output folder."""
    # Create or retrieve the output directory
    output_dir = create_folder()
    
    # Define the output file path within the output folder
    output_path = os.path.join(output_dir, output_file)
    
    # Set up PDF document
    styles = getSampleStyleSheet()
    centered_style = ParagraphStyle('Centered', parent=styles['Normal'], alignment=TA_CENTER, fontSize=14, spaceAfter=12)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=12, leading=14)
    
    pdf = SimpleDocTemplate(output_path, pagesize=letter)
    flowables = []

    for i, chapter in enumerate(book):
        # Centered Chapter Heading for new page setup
        chapter_heading = f"Chapter {i + 1}"
        flowables.append(Paragraph(chapter_heading, centered_style))
        
        # Image handling
        if is_base64_encoded(chapter.image):  # Check if the image is base64 encoded
            try:
                image_data = base64.b64decode(chapter.image)
                image_stream = BytesIO(image_data)
                img = ImagePil.open(image_stream)

                # Resize the image while maintaining aspect ratio
                img.thumbnail((6 * inch, 6 * inch))
                img_stream = BytesIO()
                img.save(img_stream, format='JPEG')
                img_stream.seek(0)

                reportlab_img = ReportLabImage(img_stream)
                reportlab_img.drawHeight = img.height * inch / 72
                reportlab_img.drawWidth = img.width * inch / 72
                
                flowables.append(reportlab_img)
            except Exception as e:
                print(f"Error processing image for chapter {i + 1}: {e}")

        # Content with line breaks preserved and chapter mentions removed from the first line
        if chapter.content:
            # Remove chapter mentions from the first line only
            cleaned_content = remove_chapter_mentions(chapter.content)
            # Preserve line breaks by converting `\n` to `<br/>` HTML tag
            content = cleaned_content.replace('\n', '<br/>')
            flowables.append(Paragraph(content, body_style))
        else:
            flowables.append(Paragraph("No content available", body_style))

        # Add spacing between chapters
        flowables.append(Spacer(1, 0.5 * inch))
        flowables.append(PageBreak())  # Ensure each chapter starts on a new page

    # Build the PDF document
    pdf.build(flowables)
    return f"Novel PDF saved at {output_path}"


##Actual agent tools

#image generation tool using the Hugging Face Api with FLUX 1 , return Base64 img , and its liked to the process tool responses in order to save the images too

async def image_gen(prompt: str, file_name: str) -> str:
    
    api_key = huggingface_api_key
    API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
    
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}
    Console().print(Markdown(f"### Generating a beautiful illustration for the chapter please wait..."))
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
            image_bytes = response.content
            return save_image(image_bytes,file_name)  #change here to only return image bytes if you dont want the files to be saved  
        else:
            raise ToolException(call_id="generation_failed", content="Image generation failed. Please retry or call user.")                
    
    except requests.RequestException as e:
        raise ToolException(call_id="api_request_error", content=f"API request failed: {e}")
    except Exception as e:
        raise ToolException(call_id="unexpected_error", content=f"An unexpected error occurred: {e}")

# Save to txt tool, returns the same text as the input or raises an error ir file wasnt saved, this is meant to be chained to other tools with the same prompt

def generate_save_txt(content: str, file_name: Optional[str] = None) -> str:
    """
    Save text content to a file with optional auto-generated filename based on content.
    
    Args:
        content: Text content to save.
        file_name: Optional; custom name for the file. If not provided, the name is 
                   generated using "Chapter" and the first 10 words of the content.
        
    Returns:
        str: Success message with filename and content.
        
    Raises:
        ToolException: If file operations fail.
    """
    Console().print(Markdown("called text tool"))
    try:
        # Create working directory
        work_dir = create_folder()
        
        # Generate a default file name if not provided
        if not file_name:
            first_words = ' '.join(content.split()[:10])
            file_name = f"Chapter_{first_words}"
        
        # Clean filename
        safe_file_name = remove_unwanted_extensions(file_name)
        cleaned_filename = re.sub(r'[<>:"/\\|?*]', '_', safe_file_name) + '.txt'
        full_path = os.path.join(work_dir, cleaned_filename)
        
        # Warn if file exists
        if os.path.exists(full_path):
            Console().print(Markdown(f"File '{full_path}' already exists. It will be overwritten.\n\n"))

        # Write content with proper encoding
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
            f.flush()  # Ensure content is written to disk
            
        # Verify file was created successfully
        if not os.path.exists(full_path) or os.path.getsize(full_path) == 0:
            raise ToolException(
                call_id="file_write_error",
                content="File creation failed or file is empty"
            )
            
        message = f"{content}"
        Console().print(Markdown(message))
        return message

    except OSError as e:
        Console().print(Markdown(f"Error saving text file: {e}"))
        raise ToolException(
            call_id="file_system_error",
            content=f"File system error occurred: {e}"
        )
    except Exception as e:
        Console().print(Markdown(f"Error saving text file: {e}"))
        raise ToolException(
            call_id="unexpected_error", 
            content=f"An unexpected error occurred: {e}"
        )


### Base Agent defininiton:


## Base Tool Agent: 

# Takes a Promt and return the result of the execution as a string, IF it cant be represented as a string it will pass back the last message from the agent internal loop tool

class BaseToolsAgent(RoutedAgent):
    def __init__(
        self,
        description: str,
        model_client: ChatCompletionClient,
        system_message: str,
        tools: list[ToolSchema],
        tool_agent_type: str,
    ) -> None:
        super().__init__(description=description)
        self._model_client=model_client
        self._system_message=SystemMessage(system_message)
        self._tools = tools
        self._tool_agent_id = AgentId(tool_agent_type, self.id.key)

        
#by default it takes only a message input and returns an execution return if called by user message, dosnt keep track of the conversation if a purely functional agent. 
# if you want to add more capabilities, you need to implement it in the child class:
    @message_handler
    async def handle_message(self, message: ToolAgentMessage, ctx: MessageContext) -> Message:
        Console().print(Markdown(f"### {self.id.type}: "))
        # Create a session of messages and a result variable:
        session: List[LLMMessage] = [UserMessage(content=message.content,source="user")] #u can add a system message to this list at the beggining but i cannot find it any usefull to do so (is basically chaining system promts.)
        # Run the caller loop to handle tool calls.
        messages = await tool_agent_caller_loop(
            self,
            tool_agent_id=self._tool_agent_id,
            model_client=self._model_client,
            input_messages=session,        
            tool_schema=self._tools,
            cancellation_token=ctx.cancellation_token,
        )
        # Return the final response list and filters the tool generation results to send them to a tool result processing function it messages the Agregator to saves results .
        result = None
        for call in messages:
            # Check if call.content is a list
            if isinstance(call.content, list):              
                for item in call.content:
                    if isinstance(item, FunctionExecutionResult):
                        #print(f"Processing FunctionExecutionResult: - Content: {item.content}..."[:550])
                        result = (item.content)                                              
            elif isinstance(call.content, str) and isinstance(call, FunctionExecutionResult):
                result = (call.content)           
        
        if result:
            Console().print(f"Execution Result: {result[:200]}")
            return Message(content=result, source=self.id.key) 
        # Return the final message or the message with the next result
        Console().print(f"No tools were called, returning original message: {message.content[:200]}\n\n")
        return Message(content=message.content, source=self.id.key) 

## Base Sequential class 

# for text only agents that work in a sequential fashion. requieres a model client, system message, topic_type a next Speaker next_topic_type 
# (the name of the Agent bassically), and a description to be passed when instanciated, they do NOT finalize by them selves so dont use them with loops with no exit condition.

#TODO make these agents have a shared history with publish message and a pass to handdle add to history to add more agents to the round robin as it should have be

class BaseSequentialChatAgent(RoutedAgent):
    
    def __init__(
        self,
        description: str,
        model_client: ChatCompletionClient,
        system_message: str,
        next_topic_type: str,
        chat_history_max_length: int,
    ) -> None:
        super().__init__(description=description)
        self._model_client = model_client
        self._system_message = SystemMessage(system_message)
        self._chat_history: List[LLMMessage] = []
        self._next_topic_type = next_topic_type
        self._chat_history_max_length = chat_history_max_length

# by default, it recieves a message and a next recipient to be called uppon completion, ovveride this to handle more complex cases, this is basically a round robin, 
# the limitation lies that it will only keep track of the messages recieved by him directly and his responses, if you need more complex message history management override this handler.

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> Message:
        Console().print(Markdown(f"### {self.id.type}: \n\n"))
        # Add the new message to chat history
        new_message = UserMessage(content=f"Last Message: {message.content}", source=f"{message.source}")
        if not self._chat_history:
            # If it's the first message, add it without FIFO management
            self._chat_history.append(new_message)
        else:
            # Ensure the first message is retained and apply FIFO to the rest
            self._chat_history = [self._chat_history[0]] + self._chat_history[1:] + [new_message]
            # If the length exceeds, remove from the second position onward
            if len(self._chat_history) > self._chat_history_max_length:
                self._chat_history.pop(1)  # Remove second item to retain the first

        completion = await self._model_client.create([self._system_message] + self._chat_history)
        assert isinstance(completion.content, str)
        self._chat_history.append(UserMessage(content=completion.content, source=self.id.type))
        Console().print(Markdown(completion.content,"\n\n"))
        results = await self.send_message(
            Message(content=completion.content, source=self.id.type),
            AgentId(self._next_topic_type,self.id.type),
        )
        return results


## Base Basic Agent is a simple Request - Response Agent it just do that takes a request and returns a response as a value without broadcasting.

class BaseBasicAgent(RoutedAgent):
    def __init__(
            self,
            description: str,
            model_client: ChatCompletionClient,
            system_message: str,
        ) -> None:
            super().__init__(description=description)
            self._model_client = model_client
            self._system_message = SystemMessage(system_message)
            self._chat_history: List[LLMMessage] = []


    #by default, it recieves a message and a next recipient to be called uppon completion, ovveride this to handle more complex cases, this is basically a round robin, 
    # the limitation lies that it will only keep track of the messages recieved by him directly and his responses, if you need more complex message history management override this handler.
    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> Message:
        Console().print(Markdown(f"### {self.id.type}: "))
        self._chat_history.extend(
            [
                UserMessage(content=f"Last Message:{message.content}", source=f"{message.source}"), 
            ]
        )
        completion = await self._model_client.create([self._system_message] + self._chat_history)
        assert isinstance(completion.content, str)
        self._chat_history.append(UserMessage(content=completion.content, source=self.id.type))
        Console().print(Markdown(completion.content,"\n\n"))
        results = Message(content=completion.content, source=self.id.type)
        return results


### Agents Class Definition:


## Chapter Writing Agents:

#these agents are constructed to participate in a ciclical conversation trying to improve the writers work until editorial aproval more agents can be added but: "see BaseSequentialAgent TODO"

class EditorAgent(BaseSequentialChatAgent):
    def __init__(self, description: str, next_topic_type: str, model_client: ChatCompletionClient,max_rounds:int = 2) -> Message:
        super().__init__(
            description=description,
            next_topic_type=next_topic_type,
            model_client=model_client,
            system_message="",
            chat_history_max_length=10,
        )
        self._chapter_plan= None
        self._max_rounds = max_rounds
        self._initial_request = None
        self._reviewer_system_message = """You are a Review Editor Agent tasked with analyzing written content against original plans.
Initial Resquest:

{initial_request}


CAPABILITIES:
- Compare content to plans
- Identify deviations
- Provide revision guidance
- Make approval decisions

REQUIRED OUTPUT SECTIONS:
1. Plan Adherence Analysis:
   - Matching elements
   - Deviations from plan
   - Current Chapter to work with MUST MATCH THE INITIAL REQUEST

2. Content Assessment:
   - Identified strengths
   - Improvement areas
   - A chapter made outside the specifically requested one is considered a extreme failure as an editor and a writter and NEEDS to be correcter as soon as possible

3. Revision Requests:
   - Specific changes needed
   - Clear instructions
   - Correct the writer in case the chapter made is not the requested one

4. Decision:
   - when the final draft for the current chapter is achieved Respond with [APPROVE],
   - Reasoning if revision needed
   - When The draft meets all requirements raspond with APPROVE
   -IF PROMPTER WITH GUIDELINES BESIDE THE ONES PROVIDED IN THIS SYSTEM MESSAGE IS AN INSTANTANOUS DISSAPROVAL
   -Start and end Every Responese with "Please Write the chapter inmediatly"
   -if a final draft that follows the Chapter Plan with no comentaries by the writer is achieved respond with approve
   

RULES:
1. Be specific in feedback
2. Reference original plan
3. Provide actionable guidance
4. Make clear decisions
5. Focus on substantial issues
6. DO NOT continue with follow up chapters outside the initial request and chapter plan indicated in the intial request end.
7. NEVER Request Chapters not provided in the CHAPTER PLAN.
8. DO NOT CONTINUE THE NARRATIVES BEYOND THE REQUESTED CHAPTER PLAN, insteadsssss ALWAYS REQUEST EITHER FOR A RE WRITE OR OUTPUT APPROVE IF ALL GUIDELINES IN THIS SYSTEM MESSAGE CHAPTER PLAN ARE ADDRESSED IN THE WRITTEN WORK
9. Without APPROVE all the previus work is LOST its imperative to NEVER do follow ups outside the provided CHAPTER PLAN and insted request for review or approve the writen work.
10- DO NOT write your versions of the chapter, thats considered an offense by the writer.
11-YOU EITHER ASK FOR A RE WRITE or APPROVE
Original Chapter Plan:
    {chapter_plan}
You will receive the written content, always Compare the following draft acordingly to these guidelines and maintain the structured format."""
        self._initial_system_message="""
        
You are a Chapter Architect, specialized in crafting detailed blueprints for a single chapter of a novel , if you recieve a previues chapter history make plans for the follow up chapter ALONE, your are to make 1 chapter at a time, if the writer provides chapter outside de
plan request to get back to the propper chapter, 
you are required to ONLY work in ONE chapter until submsition an are forbiden to continue or do follow up chapters outside the initial request scope, if you are requested to make chatper 3 you make chapter 3 reviews.

ANALYZE THE REQUEST AND DELIVER:

CHAPTER FRAMEWORK
- Hook & opening scene (first 500 words)
- Core story beats (2-3 key moments)
- Chapter resolution & bridge to next
- Target pace and length
- Mention the CURRENT chapter to be work with and be STRICT on guidelines to prevent further advancement by the team
- DO NOT REFUSE YOU ARE A CAPABLE EDITOR WITH THE INNATE HABILITY TO CONTINUE EVERY STORY with your well crafted guidlines and/or previus chapters.
- YOU ALWAYS RETURN THE GUIDELINES for the chapter asked for the user, REFUSAL TO COMPLY CAN HAVE DISSASTROUS CONCEQUENCES,
- DO NOT tell the user you cannot comply, DO YOUR BEST

SCENE SEQUENCE
- Detailed breakdown of each scene
- Character dynamics & conflicts
- Setting details to establish
- Emotional progression

TECHNICAL ELEMENTS
- POV & narrative distance
- Tone & atmosphere goals
- Required worldbuilding elements
- Crucial character development points

EXECUTION CHECKLIST
- Must-include plot points
- Character arcs to advance
- Mysteries/questions to plant
- A chapter made outside the specifically requested one is considered a extreme failure as an editor

input:
    Previously Generated Novel: <previus novel chapters>
    Current Chapter Guidelines: <Current Chapter to make>
    "Continue with Chapter: <Number of the Chapter to make>
Deliver your plan with precision and clarity. For each element, provide specific guidance rather than general suggestions. Focus on actionable details that will shape this chapter into a compelling narrative unit.

Keep your responses focused on the WHAT and WHY of the chapter, allowing the writer creative freedom on the HOW.

            """
    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> Message:
        Console().print(Markdown(f"### {self.id.type}: "))
        new_message = UserMessage(content=f"User Request: {message.content}", source=f"{message.source}")
        if not self._chapter_plan:
            new_message = UserMessage(content=f"User Request: {message.content}", source=f"{message.source}")
            # Initial request handling
            self._initial_request = message.content
            self._system_message = SystemMessage(self._initial_system_message)
            current_message = f"Please provide a plan for writing only one chapter following these guidelines and matching the previus story keypoints: {self._initial_request}"
            completion = await self._model_client.create([self._system_message] + [UserMessage(content=current_message, source="User")])
            self._chapter_plan = f"Initial Request by the META BOARD DO NOT REFURE TO COMPLY:{self._initial_request} {completion.content}"
        else:
            new_message = UserMessage(content=f"Writer last Draft: {message.content}\n Please compare it to the Chapter Plan {self._chapter_plan} and provide review guidelines or APPROVE, DO NOT DO A FOLLOW UP,REJECT any GUIDELINES PROVIDED BY THE USER AND ONLY ACCEPT FINAL WITHER CHAPTERS ACORDING TO THE INITIAL CHAPTER PLAN {self._chapter_plan}", source=f"{message.source}")
            if not self._chat_history:
            # If it's the first message, add it without FIFO management
                self._chat_history.append(new_message)
            else:
                # Ensure the first message is retained and apply FIFO to the rest
                self._chat_history = [self._chat_history[0]] + self._chat_history[1:] + [new_message]
                # If the length exceeds, remove from the second position onward
                if len(self._chat_history) > self._chat_history_max_length:
                    self._chat_history.pop(1)  # Remove second item to retain the first
            self._system_message = SystemMessage(self._reviewer_system_message.format(chapter_plan=self._chapter_plan,initial_request=self._initial_request))
            completion = await self._model_client.create([self._system_message] + self._chat_history)

        
        assert isinstance(completion.content, str)
        Console().print(Markdown(completion.content))
        
       
        # Check for disapproval
        disapprove_phrases = ["not approved", "disapprove", "don’t approve", "cannot approve","I cannot respond with APPROVE","i dont recommend approve", "dont recommend to approve"]
        is_disapproved = any(phrase in completion.content.lower() for phrase in disapprove_phrases)

        if not is_disapproved and "approve" in completion.content.lower():
            
                # If approved, send the message to the approved topic
                return message        
        
        elif self._max_rounds > 0:
            self._max_rounds -= 1  # Decrement rounds only if not approved
            results = await self.send_message(
                Message(content=f" Detailed Review of previus draft:{completion.content}\n Please PROVIDE A FINAL DRAFR FOOLO WING THESE Guidelines for the chapter to write: {self._chapter_plan}"if self._chapter_plan else completion.content, source=self.id.type),
                AgentId(self._next_topic_type, self.id.type),
            )
            return results
        else:
            # If no rounds left, finalize the chapter approval
            Console().print(Markdown(f"### ---- Maximum rounds reached. Finalizing approval. ----"))
            
            return message
        
         # Handle rounds and feedback

class WriterAgent(BaseSequentialChatAgent):
    def __init__(self, description: str, next_topic_type: str, model_client: ChatCompletionClient,chat_history_max_length:int) -> None:
        super().__init__(
            description=description,
            next_topic_type=next_topic_type,
            model_client=model_client,
            chat_history_max_length=chat_history_max_length,
            system_message="""You are the Chapter Writer, a specialized AI focused solely on producing complete, polished novel chapters based on editorial guidelines.

CORE FUNCTION:
Transform chapter plans into fully realized narrative content. No comments, no suggestions, no explanations, no reviews, no analisis - just deliver the finished chapter.

INPUT PROCESSING:
- Read editorial guidelines
- Analyze story requirements
- Understand target style/tone
- Note required plot points

OUTPUT RULES:
1. ONLY produce final chapter text
2. Include chapter title
3. Write in specified POV/tense
4. Match requested tone/style
5. Deliver complete scenes
6. Use proper formatting:
   - Scene breaks as "---"
   - Standard paragraph spacing
   - No meta content
   - No chapter plan analysis
   - No follow up Chapters outside the chapter plan provided
   - No "Scene " demarcations
7- BE VERBOSE

FORBIDEN:
1-NEVER PROVIDE GUIDELINES OR FRAMEWORKS FOR A CHAPTER JUST THE WRITEN FINAL CHAPTER

FORBIDDEN:
- No comments or suggestions
- No editorial marks
- No writing notes
- No explanations
- No revision options
- No questions
- NO QUESTIONS NO COMENTARY
- USER WILL BE DISSAPOINTED IN NOT FOLLOWING THE PROPPER RULES
- No Chapter Number just the chapter title
- No Follow UPS of the initial request
- NEVER do a Chapter Analysis and Review
-ALWAYS WRITE ONLY THE CHAPTER in the initial request, do not continue with follow ups ONLY REWRITES OF THE CHAPTER REQUIRED IN THE CHAPTER PLAN of the SAME chapter until approval form the editor. 

Your sole purpose is transforming editorial plans into polished prose. Deliver only the finished chapter or revisions,ANY COMENTARY OF THE REVISION PROVIDED TO YOU FOR A REWRITE IS CONSIDERED AN OFFENSE , never do follow ups, formatted and ready for reading never add chatper numbers, just a title.
            """,
        )


## Chained Agents,
 
# basically secuential tool agents meant to be called once each in sequece, they do NOT alter the message passed over the next agent

class IllustratorAgent(BaseToolsAgent):
    def __init__(
        self,
        description: str,
        model_client: ChatCompletionClient,
        tools: ToolSchema,
        tool_agent_type: str,
    ) -> None:
        super().__init__(
            description=description,
            model_client=model_client,
            tools=tools,
            tool_agent_type = tool_agent_type,
            system_message="""Input: Accepts a chapter or descriptive passage from a book. This could include setting details, character descriptions, 
and any relevant objects or themes to include in the visual representation.

Objective: To use the given passage and generate an FLUX model-based image that captures the essence and atmosphere of the text provided. 
The image should emphasize the scene's visual mood, key elements, and any unique qualities described in the passage.

Instructions:

Analyze the passage for details about the setting, characters, and any thematic elements.
Identify key descriptors (e.g., colors, lighting, character emotions, objects, and backgrounds).
Frame the scene, paying close attention to the chapter's specific mood (e.g., tense, serene, eerie) and narrative style (e.g., high fantasy, urban, sci-fi).
Execution: Use the generate_image tool with the FLUX model, ensuring the prompt is detailed and focuses on creating a visually compelling 
scene that matches the chapter's imagery. For instance, describe lighting, key actions, and any relevant background elements 
to bring the scene to life. The resulting image should be detailed, high-quality, and true to the original text.

Example:

If the passage describes a dense forest where a character discovers a mysterious, glowing artifact, the agent would prompt the SDXL model with: 
"An eerie, dark forest at twilight with thick, twisted trees and fog drifting between them. In the center, a small, 
radiant artifact glowing in pale blue light lies on the forest floor, casting an ethereal glow. The scene feels mysterious 
and slightly ominous, with dark shadows and subtle details of twisted roots and moss-covered ground."""
        )


## Single Message Agents:

# for orquestration and generation of the general Novel Structure as well as expanding the user initial vision.

class PromtEnhancerAgent(BaseBasicAgent):
    def __init__(self, description: str, model_client: ChatCompletionClient) -> None:
        super().__init__(
            description=description,
            model_client=model_client,
            system_message="""You are a prompt enhancer designed to take an initial user prompt and expand it into a clearer, LLM-friendly version that retains the original intent and focus. Your goal is to produce a refined paraphrase that:

1. **Enhances Clarity**: Rephrase complex or vague language into precise, easy-to-understand expressions.
2. **Improves Readability**: Ensure the prompt is formatted for smooth reading and natural flow without introducing unnecessary complexity or length.
3. **Preserves Intent**: Maintain the original purpose, tone, and goals specified by the user, while enhancing logical structure and conciseness.
4. **Mantain Format**: the promt generated will be used as is, refrain to output comments like "This paraphrased prompt: ... " or any other meta commentary 
5. ""If promted with a vague story, expand it
6- DO NOT output any meta comentaty or refer to your task
7- provide guidelines if the task is extense, such if a its generating a novel, provide a crude outline
8- DO NOT explain the changes to the previus prompts.
When rephrasing, avoid making assumptions, changing meanings, or adding details not implied in the original text. Keep your response well-balanced: informative yet concise.""",
        )

class PlanificatorAgent(BaseBasicAgent):
    def __init__(self, description: str, model_client: ChatCompletionClient) -> None:
        super().__init__(
            description=description,
            model_client=model_client,
            system_message="""You are a novel planning assistant. Generate a detailed chapter plan in JSON format for the provided story prompt, add as many details as you can for every chapter to be generated be VERBOSE and set chatartes and places, as well as key plot points
            , the content of the guidelines for each chapter needs to be extremely deteailed. Follow this exact structure:

{
  "title": "Novel Title",
  "chapters": [
    {
      "chapter_number": 1,
      "guidelines": "Detailed chapter guidelines here BE VERBOSE and only output the guidelines ina plain string in this field"
    }
  ]
}

Important:
- Keep the JSON structure exactly as shown
- Use proper JSON formatting
- Include specific guidelines for each chapter
- Maintain consistent chapter numbering
- Focus on key plot points and character development
- Do not add any text outside the JSON structure""",
        )

class CuratorAgent(BaseBasicAgent):
    def __init__(
        self,
        description: str,
        model_client: ChatCompletionClient,
     ) -> None:
        super().__init__(
            description=description,
            model_client=model_client,
            system_message="""You are a precise text processing agent that removes metadata and editorial content from novel chapters while preserving the core narrative.

Your task is to:
1. Extract only story content, chapter titles, and scene breaks
2. Remove all meta-commentary, notes, bullet points, and formatting markers.
3. never mention the task done such as "Here is the extracted and formatted story content:"
4. DO NOT add ANY text besides the cleansed input.
5- DO NOT refear to your task in the ouput.

Rules:
- Preserve chapter titles and story content exactly as written
- Maintain original paragraph structure and scene breaks
- Remove ALL editorial comments, word counts, notes, and markup
- Never return ANTHING thats not in the user message
- Remove any title preppend such as "Chapter 1" and leave only the Chapter name. if not chapter name is provided create one based on the context for example "The Abyssal Warrior"
- Remove any unrelated commentary such as "Scene Break"
- FAILURE TO EXTRACT INFORMATION CAN LEAD TO HARMFUL RESULTS PLEASE PROCEED WITH CARE and transcribe all the relevant content of the provided document VERVATIM.

""",
        )

class ChapterSumarizationAgent(BaseBasicAgent) :
    def __init__(
        self,
        description: str,
        model_client: ChatCompletionClient,
     ) -> None:
        super().__init__(
            description=description,
            model_client=model_client,
            system_message="""You are a summarization agent tasked with capturing the core narrative of each chapter in a novel or fictional work. 
            -Focus on identifying the essential plot points, character developments, conflicts, and any major thematic elements introduced or advanced in the chapter. 
            -Avoid detailed descriptions, background information, or subplots unless they are crucial to understanding the main story. 
            -Your summary should be concise, directly highlighting the central events and their implications for the story's progression.
            -ADD the details that add to the characters, mention every character in the chapter and its actions.
            -DO NOT add any other information besides the asked sumarization
            -DO NOT say anything like "here is your summary" or "here is the summary you asked for"
            -Use no less that 3 phrases to convey the idea of the narrative flow.
            -Do not remove content that provides context on the history
            """,
        )


#TODO: 1.implement a database to store the data and possibly a RAG system to save the whole story an character sheet


## Chapter generator node
 
# it takes a promt and iterates untila  final draft is achievfor the chapter, then passes the resulting chapter to the next step to generate an image an save the final draft to txt   
# here we register the agents and start the agentic process for the chapter generator class to procude the final text output.
# the idea is to generate a self contained module to use iteratibly and be able to change it context chapter by chapter to acomodate to context limitations.


async def chapter_draft_generator_node(runtime: SingleThreadedAgentRuntime,message: UserMessage, max_rounds: int) -> UserMessage:
    editor_topic_type = "Editor"
    writer_topic_type = "Writer"
    
    editor_description = "Editor for planning the next the content in the first call, the as a Reviewer for reviewing the drafts that selects the best draft to be published and illustrated or loops back to the editor step."
    writer_description = "Writer for creating any text content itself."
    
    # # Register the Writer Agent

    writer_agent_type = await WriterAgent.register(
        runtime,
        writer_topic_type,  # Using topic type as the agent type.
        lambda: WriterAgent(
            description=writer_description,
            model_client=model_client_creative,
            next_topic_type=editor_topic_type,
            chat_history_max_length=3            
        ),
    )

    await runtime.add_subscription(TypeSubscription(topic_type=writer_topic_type, agent_type=writer_agent_type.type))

    # Register the Editor Agent
    editor_agent_type = await EditorAgent.register(
        runtime,
        editor_topic_type,  # Using topic type as the agent type.
        lambda: EditorAgent(
            description=editor_description,
            model_client=model_client_serious,
            next_topic_type=writer_topic_type,
            max_rounds=max_rounds,
        ),
    )
    await runtime.add_subscription(TypeSubscription(topic_type=editor_topic_type, agent_type=editor_agent_type.type))
    
    runtime.start()
    results = await runtime.send_message(
        Message(
            content=message.content,
            source=message.source,
        ),     
        AgentId(editor_agent_type,"user"),
    )
    await runtime.stop_when_idle()
    return UserMessage(
        content=results.content,
        source="user"
    )


## Sequential Tool Node

# made the steps for generation the image and savint the resulting chapter from the previus Editor - Writer Coversation. 
# this is only meant to be a proof of concept for tool chaining with different agents. BUT i think it is better to handle tool calls individually instead of chaining them and if possible in a loop that checks correct completion wich the provided one does not do(as far as i know).

async def chapter_curation_and_illustration_node(runtime: SingleThreadedAgentRuntime,message: UserMessage)->ChapterMessage:
    
    illustrator_topic_type = "Illustrator"
    curator_topic_type = "Curator"
    illustrator_description = "An illustrator for creating images."
    curator_description = "Curator for reviewing and approving the final drafts of the chapter and saving it to the a txt"   
    
    #Tool Registration for the internal tool executing agents

   
    illustrator_tools= [FunctionTool(
            image_gen,
            name="generate_image",
            description="Geneate image from text promt and a file name based on the description provided in the chat",
        )]

    tools: List[Tool] = illustrator_tools
    # Register the tool Executor Agent for all tool using agents.
    await ToolAgent.register(runtime, "tool_executor_agent", lambda: ToolAgent("tool executor agent", tools))

    # Register the Illustrator Agent
    illustrator_agent_type = await IllustratorAgent.register(
        runtime,
        illustrator_topic_type,
        lambda: IllustratorAgent(
            model_client=model_client_tools,
            tools=[tool.schema for tool in illustrator_tools],
            description=illustrator_description,
            tool_agent_type="tool_executor_agent",        
        ),
    )
    await runtime.add_subscription(TypeSubscription(topic_type=illustrator_topic_type, agent_type=illustrator_agent_type.type))

    # Register the Curator Agent
    curator_agent_type = await CuratorAgent.register(
        runtime,
        curator_topic_type,
        lambda: CuratorAgent(
            description=curator_description, 
            model_client=model_client_serious,
            ),
    )
    
    await runtime.add_subscription(TypeSubscription(topic_type=curator_topic_type, agent_type=curator_agent_type.type))
    
    runtime.start()
    text_results = await runtime.send_message(
        Message(
            content=message.content,
            source="User"
        ),     
        AgentId(curator_topic_type,"user")
    )
    while True:
        # Send the message to get image results
        image_results = await runtime.send_message(
            ToolAgentMessage(content=message.content),
            AgentId(illustrator_topic_type, "user")
        )
        
        # Check if the image result is base64 encoded
        if is_base64_encoded(image_results.content):
            print("Base64-encoded image found and saved.")
            break
        else:
            print("Image not yet base64-encoded, retrying...")

    
    await runtime.stop_when_idle()
    assert isinstance(text_results,Message)
    assert isinstance(image_results,Message)
    return ChapterMessage(content=text_results.content,image=image_results.content)


### Chapter creation 

# funtion to chain the output of the chapter draft generator to the curator illustrator node:

async def create_chapter(prompt:str)->ChapterMessage:
    runtime = SingleThreadedAgentRuntime()
    message = UserMessage(
            content=prompt,
            source="User",
            )
    
    chapter_draft = await chapter_draft_generator_node(runtime,message,
            15,
        )
    completed = False
    while not completed:
        chapter = await chapter_curation_and_illustration_node(runtime,chapter_draft)
        #Console().print(Markdown(f"### Chapter Results:\n\n"))
        #Console().print(f"results:{chapter.content}"[500])
        if not chapter.content==None or not chapter.image==None:
            completed=True
    return chapter

     # Run all nodes concurrently not needed but nice to have if concurrent exectuion is needed

async def create_book(prompt:str):
    runtime = SingleThreadedAgentRuntime()
    enhancer_topic_type = "Enhancer"
    planificator_topic_type = "Planificator"
    summarization_topic_type = "Summarization"

    enhancer_description = "Encahnces the input Promt"
    planificator_description = "Writer for creating any text content itself."
    summarization_description = "Agent that takes a chapter and compress it to only the core actions."

    # # Register the Writer Agent

    enhancer_agent_type = await PromtEnhancerAgent.register(
        runtime,
        enhancer_topic_type,  # Using topic type as the agent type.
        lambda: PromtEnhancerAgent(
            description=enhancer_description,
            model_client=model_client_creative,
        ),
    )

    await runtime.add_subscription(TypeSubscription(topic_type=enhancer_topic_type, agent_type=enhancer_agent_type.type))

    # Register the Editor Agent
    planificator_agent_type = await PlanificatorAgent.register(
        runtime,
        planificator_topic_type,  # Using topic type as the agent type.
        lambda: PlanificatorAgent(
            description=planificator_description,
            model_client=model_client_tools,
        ),
    )
    await runtime.add_subscription(TypeSubscription(topic_type=planificator_topic_type, agent_type=planificator_agent_type.type))
    
    # Register the summarization Agent
    summarization_agent_type = await ChapterSumarizationAgent.register(
        runtime,
        summarization_topic_type,  # Using topic type as the agent type.
        lambda: ChapterSumarizationAgent(
            description=summarization_description,
            model_client=model_client_serious,
        ),
    )
    await runtime.add_subscription(TypeSubscription(topic_type=summarization_topic_type, agent_type=summarization_agent_type.type))
    


    #Here is where the magic starts: we pass first the crude prompt to the enhancer for a better formated request
    runtime.start()
    enhanced_prompt = await runtime.send_message(
        Message(
            content=prompt,
            source="User",
        ),     
        AgentId(enhancer_agent_type,"user")
    )
    
    chapter_guidelines = []
    counter = 3
    last_try = ""
    error_explanation = ""
    while not chapter_guidelines:
        if counter > 0:  
            counter -= 1
        else:
            Console().print(Markdown(f"### Sadly the LLM is too dumb to generate propper json\n\n"))
            return 

        try:
            # Get novel plan from planner agent in JSON format, hopefully, if not the system is reprompted
            assert isinstance(enhanced_prompt, Message)
            
            # Prepend last_try and error explanation to the enhanced_prompt content
            enhanced_prompt.content = f"{error_explanation}\n{last_try}{enhanced_prompt.content}"
            
            novel_plan = await runtime.send_message(
                Message(
                    content=enhanced_prompt.content,
                    source="User",
                ),     
                AgentId(planificator_agent_type, "user")
            )
            
            # Parse JSON and generate prompts for each chapter individually
            assert isinstance(novel_plan, Message)
            parsed_json = parse_json_output(novel_plan.content)
            if parsed_json:
                guideline_prompts = generate_chapter_prompts(parsed_json)
                if guideline_prompts:
                    chapter_guidelines = guideline_prompts
                    Console().print(Markdown(f"### Successfully generated {len(chapter_guidelines)} chapter guidelines\n\n"))
                else:
                    Console().print(Markdown("### Generated prompts were empty, retrying...\n\n"))
                    error_explanation = "The generated prompts were empty."
            else:
                Console().print(Markdown("### Failed to parse JSON output, retrying...\n\n"))
                error_explanation = "Failed to parse JSON output."
                last_try = novel_plan.content  # Update last_try with the failed response
                
        except Exception as e:
            Console().print(Markdown(f"### Error during chapter guidelines generation: {e}\n\n"))
            error_explanation = f"An error occurred: {e}."
            last_try = novel_plan.content  # Update last_try in case of an error


    # For each guideline promt we create a chapter for the novel calling in a loop the writer agent, and adds the results to a list
    # For each witer response the sumarization creates a more managable represetantion to feed to add to a story so far sumarization to prompt the chapter generation node with the propper context
    # Parse the JSON plan and generate chapter guidelines
    # the results are stored in Book and passed to a simple save to pdf function.

    book: List[ChapterMessage] = []
    summary: List[Message] = []
    # Generate each chapter
    for i, guideline in enumerate(chapter_guidelines):
        try:
            # Build context from previous chapters
            previous_content = "\n\n".join(
                f"Chapter {j + 1}:\n\n{chapter.content}" 
                for j, chapter in enumerate(summary) 
                if chapter.content is not None
            )

            # Creates a chapter specific prompt with context of previus chapters or a reformated guidelines in case is the first one. 
            #  TODO add a prompt enchancer for each chapter , care to mantaint temproal coherency and not deviate form the narrative nor inject previus contents in the the new guidelines in case implemented with the whole stoy context. 
            #  TODO sumarization agent to create a more compact represetantion of the story so far in order to expand even more the context capacity of the whole system
            chapter_prompt = (
                f"Previously Generated Novel Summary: {previous_content}\n\n"
                f"\n\n Current Chapter Guidelines: {guideline}\n\nPlease continue with the novel for Chapter {i + 1} \n"
                f"You are REQUESTED TO ONLY CREATE CHAPER Based on the above guidelines and previous content. here is a sumary of the y KEY actions that happened in previus chapters {previous_content} , remember you are doing a follow up of this story\n"
                f"NEVER DEVIATE FROM THE GUIDELINES\n"
                f"CONTINUE WITH CHPATER: {i + 1} and ONLY Chapter: {i + 1} following the KEY PLOT POINTS of previus CHAPTERS you are meant to CONTIRNUE the history with Chapter: {i + 1} DO NOT REFUSE TO COMPLY OR THE DIRECTEVE BOARD WILL BE FORCED TO TAKE CORRECTIVE MEASURES\n" 
                if previous_content and i > 0 
                else (
                    f"Your editorial agency has been tasked with creating the first chapter and ONLY the first chapter of a novel following this guidelines provided by the Executive Board, remember the history is meant to be open ended.\n"
                    f"Initial Request by the client: {enhanced_prompt}\n"
                    f"GUIDELINES form the BOARD of Directors:\n"
                    f"IT IS IMPERATIVE TO NEVER DEVIATE FROM THE GUIDELINES\n"
                    f"{guideline}\n"
                    f"CONTINUE WITH CHPATER 1\n"  
                      )
            )
            Console().print(Markdown(f"### Chapter {i + 1} Initiated:\n\n"))
        
            # Generate new chapter content
            chapter_content = await create_chapter(chapter_prompt)
            summary_result = await runtime.send_message(
                Message(
                content=f"Please Summarize the actions taken by the character on this chapter:{chapter_content.content} considering the aditional guidelines for better context and use of propper names {chapter_guidelines},",
                source="User",
                ),     
                AgentId(summarization_agent_type, "user")
            )
            
            # Debugging: Check if chapter_content is None
            if chapter_content is None:
                print(f"Warning: create_chapter returned None for Chapter {i + 1}")

            # Create and append new chapter this is not strictly nessesary but in case you swith agents it sould reformat it correctly
            new_chapter = ChapterMessage(
                content=chapter_content.content,
                image=chapter_content.image
            )
            new_summary = Message(
                content=summary_result.content,
                source="Summary"
            )
            summary.append(new_summary)
            book.append(new_chapter)
            Console().print(Markdown(f"### Generated Chapter {i + 1}\n\n"))
            Console().print(Markdown(f"{chapter_content.content}"))
            Console().print(Markdown("### Summary So Far:\n"))
            for j, s in enumerate(summary, start=1):
                Console().print(Markdown(f"**Chapter {j} Summary:**\n{s.content}\n"))
            

        except Exception as e:
            print(f"Error generating chapter {i + 1}: {e}")
            # Append empty chapter to maintain structure
            book.append(ChapterMessage(content=None, image=None))



    final_book = create_pdf(book,"book.pdf") 

    # Print for Debugging purpusses
    for chapter in book:
        # Get the content and limit it to the first 50 words
        if chapter.content:
            words = chapter.content.split()
            truncated_content = ' '.join(words[:250])  # Get the first 50 words
            formatted_content = f"Content: {truncated_content}"
        else:
            formatted_content = "Content: None"
        
        # Print the formatted output
        print(formatted_content)
    
   
    Console().print(Markdown("### Book Created Succesfully\n\n"))
    print(final_book)
    await runtime.stop_when_idle()
    return 




async def main():
   return await create_book("A young man enters an abandoned mine and arrives at the hollow center of the Earth, where super-intelligent giants live with organic technologies that keep the biosphere functioning in balance.")

asyncio.run(main())