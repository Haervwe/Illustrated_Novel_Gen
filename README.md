# Illustrated Novel Generation with Autogen Core

## An experiment in orchestrating local LLM agents for creative workflows

This project explores the capabilities of agentic workflows for creative writing, specifically illustrated novel generation. It leverages the Autogen Core framework to coordinate multiple local LLM agents, each with a specialized role in the process.

**Key Features:**

* **Local LLM execution:** Utilizes Ollama to run large language models locally, offering flexibility for integration with other OpenAI API compliant LLM providers.
* **Agentic workflow:** Employs Autogen Core 0.4 to manage interactions between specialized agents, enhancing collaboration and efficiency.
* **Multi-model inference:**  Different local LLMs can be used for various agent roles, allowing for diverse and tailored outputs based on model strengths.
* **Integrated image generation:**  Generates images for each chapter using the Hugging Face API and FLUX model, adding a visual dimension to the narrative.
* **Automated PDF creation:** Compiles chapters and images into a professionally formatted PDF book, ready for sharing or printing.


## System Architecture

The novel generation process follows a structured workflow:

1. **User Input:** The user provides an initial story prompt, sparking the creative process.

2. **Prompt Enhancement:** The `PromptEnhancerAgent` refines the user prompt, improving clarity, readability, and structure for optimal LLM processing.

3. **Novel Planning:** The `PlanificatorAgent` generates a comprehensive JSON-formatted plan outlining detailed guidelines for each chapter, including plot points, character arcs, and setting details.

4. **Chapter Generation Loop:** This iterative loop is the heart of the system, crafting each chapter with precision:
   - **Chapter Guideline Composer:**  Dynamically prepares a specific prompt for the current chapter, incorporating context and summaries from previous chapters to maintain narrative coherence.
   - **Editor-Writer Loop:** The `EditorAgent` and `WriterAgent` engage in a collaborative refinement process. The `WriterAgent` drafts the chapter based on guidelines, while the `EditorAgent` provides feedback and ensures adherence to the plan. This loop continues until the chapter reaches an acceptable quality.
   - **Curation:** The `CuratorAgent` meticulously removes any meta-commentary or artifacts from the final chapter text, ensuring a polished and immersive reading experience.
   - **Illustration:** The `IllustratorAgent` generates a visually compelling image based on the chapter content, capturing the essence of the narrative and enriching the reader's imagination.
   - **Summarization:** The `ChapterSummarizationAgent` creates a concise summary of the chapter, extracting key events and character developments to provide context for subsequent chapters and maintain narrative flow.

5. **PDF Compilation:**  Once all chapters are complete, the system automatically compiles them, along with their corresponding illustrations, into a beautifully formatted PDF book. This final output is ready for distribution and enjoyment.


## Flow Diagram

![Untitled-2024-10-31-1933](https://github.com/user-attachments/assets/7c157d66-51bf-4bee-983f-b46236a04734)



## Agents

Each agent plays a crucial role in the novel generation process:

* **PromptEnhancerAgent:**  Refines and optimizes the initial user prompt for clarity and LLM compatibility.
* **PlanificatorAgent:** Generates a detailed, structured chapter plan in JSON format, serving as a roadmap for the narrative.
* **EditorAgent:**  Provides constructive feedback, guidance, and approval on chapter drafts, ensuring quality and adherence to the plan.
* **WriterAgent:**  Crafts the chapter content, translating guidelines and feedback into engaging prose.
* **CuratorAgent:** Cleans and formats the final chapter text, removing any extraneous information and ensuring a polished reading experience.
* **IllustratorAgent:**  Generates evocative images for each chapter, enhancing the visual appeal and immersive quality of the story using function tools to call the Hugging Face API using FLUX.
* **ChapterSummarizationAgent:**  Summarizes chapters concisely, providing essential context for subsequent chapters and aiding in narrative coherence this is a core role since provides a good way to manage context size withut sacrificing previous narrative arcs.


## Tools

The system utilizes powerful tools for image generation and PDF creation:

* **Hugging Face API:**  Enables access to the FLUX model for generating high-quality illustrations.
* **ReportLab:**  Provides the functionality for compiling chapters and images into a professional-looking PDF book.


## Results and Observations

* **Qualitative Assessment:**  The quality of generated novels can vary significantly depending on the specific LLMs used, prompt engineering, and the complexity of the narrative. Longer context chains and intricate factual scenarios can pose challenges to maintaining coherence and consistency.

* **Challenges:**  Achieving optimal performance requires careful selection and configuration of LLMs. Smaller fine-tuned models (around 3.2B parameters) can produce good results, but larger models (at least 8B parameters) with strong reasoning capabilities are recommended for more complex and extensive novels. Effective prompt chaining and context management are crucial for consistent and coherent output.

* **Future Work:**  Future development aims to enhance the system's capabilities by incorporating RAG (Retrieval-Augmented Generation) agents to access and utilize external knowledge sources. This will enable the creation of richer and more factually grounded narratives. Additionally, generating background lore and detailed character sheets will pre-fill the narrative with compelling elements, further enhancing the creative process. Further research and testing will focus on refining summarization processes and improving the overall quality and consistency of chapter generation.

## Educational Focus

This project is designed with an educational focus, aiming to provide a clear and accessible example of how to utilize the Autogen Core framework for creative applications. The codebase is extensively commented to:

* **Explain the purpose and functionality of each agent and component.**
* **Illustrate the flow of information and interactions within the agentic workflow.**
* **Highlight key concepts and techniques in prompt engineering and LLM orchestration.**
* **Encourage experimentation and modification of the code to explore different creative possibilities.**

By studying and experimenting with this project, users can gain valuable insights into the potential of agentic AI for creative tasks and learn how to apply the Autogen Core framework to their own projects.

## Running the Code

TODO

Run the Application: Follow the specific instructions within the codebase to start the application and provide your inspiring story prompt. Let the creative exploration begin!

