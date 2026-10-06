from crewai import Agent, LLM, Task, Crew, Process
from crewai_tools import ArxivPaperTool
import os
from dotenv import load_dotenv
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
load_dotenv()

# Initialize Gemini explicitly
gemini_llm = LLM(
    model="gemini/gemini-3.5-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY")
)

arxiv_tool = ArxivPaperTool(download_pdfs=False, max_results=5)

## Implement embedding for using it in chromadb

google_ef=embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    model_name="gemini-embedding-001",
    api_key=os.getenv("GEMINI_API_KEY")
)

# Use Persistent ChromaDB client to store the cache on your disk
chroma_client=chromadb.PersistentClient(path="./semantic_cache_db")
cache_collection=chroma_client.get_or_create_collection(
    name="research_synthesis_cache", 
    embedding_function=google_ef
)

# Agent 1: The Scout
literature_scout = Agent(
    role="Academic Literature Scout",
    goal="Discover and summarize the most relevant, high-impact academic papers regarding a specific hypothesis.",
    backstory="You are a meticulous research librarian. You excel at querying academic databases, filtering out irrelevant studies, and identifying foundational papers.",
    tools=[arxiv_tool],
    llm=gemini_llm,
    verbose=True
)

# Agent 2: The Critic
methodology_critic = Agent(
    role="Methodology Critic",
    goal="Evaluate the scientific rigor, dataset validity, and statistical significance of the retrieved papers.",
    backstory="You are a ruthless peer-reviewer. You do not care about the hype of a paper; you only care about sample sizes, p-values, and potential methodological flaws.",
    llm=gemini_llm,
    verbose=True
)

# Agent 3: The Synthesizer
synthesis_writer = Agent(
    role="Research Synthesis Writer",
    goal="Draft a comprehensive, critically evaluated literature review combining the scout's findings and the critic's evaluations.",
    backstory="You are an elite academic author. You weave disparate studies into a coherent narrative, explicitly highlighting gaps in the current research.",
    llm=gemini_llm,
    verbose=True
)

# If you dont want to add tasks here in the same file of agents.py then you need to add the agents in .jsonc file 

"""
    the code for the .jsonc will be - 
    {
  "name": "Research Crew",
  "agents": ["researcher"],
  "tasks": [
    {
      "name": "research_task",
      "description": "Conduct thorough research about {topic}. Use web search to find recent, credible information.",
      "expected_output": "A markdown report with clear sections: key trends, notable tools or companies, and implications. Aim for 800-1200 words. No fenced code blocks around the whole document.",
      "agent": "researcher",
      "output_file": "output/report.md",
      "markdown": true
    }
  ],
  "process": "sequential",
  "verbose": true
}

"""

# But if you want to add tasks here then you can use it like it below


############ TASKs ################


# Task 1: Literature Scouting
scouting_task = Task(
    description=(
        "Search arXiv for 5 foundational academic papers related to the hypothesis: '{hypothesis}'. "
        "Extract the title, authors, abstract, and core claims of each paper. "
        "Do not download PDFs; rely on the abstract and metadata."
    ),
    expected_output="A structured list of 5 papers with their titles, authors, and summarized core claims.",
    agent=literature_scout
)

# Task 2: Methodology Critique
critique_task = Task(
    description=(
        "Review the core claims and abstracts provided by the literature scout. "
        "Identify potential methodological flaws, limited sample sizes, or dataset biases in each study. "
        "Assign a confidence score (1-10) to each paper's findings based on scientific rigor."
    ),
    expected_output="A detailed critique for each paper, including identified biases, methodological gaps, and a final confidence score.",
    agent=methodology_critic
)

# Task 3: Synthesis Writing (Output to Markdown)
synthesis_task = Task(
    description=(
        "Combine the scout's paper summaries and the critic's evaluations into a cohesive literature review. "
        "Highlight contradictions between studies and identify the most critical gaps in the current research landscape. "
        "Format the final document professionally."
    ),
    expected_output="A comprehensive, multi-section academic literature review.",
    agent=synthesis_writer,
    markdown=True, # Enables automatic markdown formatting instructions
    output_file="research_review.md" # Automatically creates the file with the task's final output
)

# Orchestrate the Crew with Memory
research_crew = Crew(
    agents=[literature_scout, methodology_critic, synthesis_writer],
    tasks=[scouting_task, critique_task, synthesis_task],
    process=Process.sequential,  #  this means the functions will be called as you defined , there also is .hierarchial where agent decides which function to choose
    memory=True, # Enables shared short-term, long-term, and entity memory across the crew
    embedder={
        "provider": "google-generativeai",
        "config": {
            "model_name": "gemini-embedding-001",
            "api_key": os.getenv("GEMINI_API_KEY")
        }
    },
    verbose=True,# have a detailed logging of what agent is doing
    tracing=True
)

# if __name__=="__main__":
def run_research(target_hypothesis):
    import uuid
    # Use this bvelow hypothesis for first ryun , to store it in cache
    #target_hypothesis = "The impact of Retrieval-Augmented Generation (RAG) on reducing hallucination rates in medical diagnostics."

    #Use this below hypothesis for next run , to test if cache is working fine or not 
    # target_hypothesis = "How RAG architecture mitigates factual errors in clinical AI systems."
    #target_hypothesis = "Does using RAG decrease hallucinations for AI in healthcare?"
    print(f"Initiating research on: {target_hypothesis}")

    # Calculate the distance of the query while searching in cache , using dense vector search
    # We are use 0.25 for some rephrasing correction , for strict search use 0.0

    cache_results = cache_collection.query(
        query_texts=[target_hypothesis],
        n_results=1
    )

    if len(cache_results['distances'][0]) > 0 and cache_results['distances'][0][0] < 0.25:
        print("⚡ CACHE HIT")
        return cache_results['documents'][0][0], True

    #USE This BLOCK OF CODE FOr __NAME=="__main__"
    

    else:
        # 3. Cache MISS: Run the full expensive multi-agent pipeline
        print("\n🔍 CACHE MISS: No similar research found. Waking up the agent crew...")

        # we use .kickoff() to run a whole crew agent pipepline at once
        result = research_crew.kickoff(inputs={"hypothesis": target_hypothesis})
        
        # Extract the final markdown string from the Crew Output
        final_markdown = result.raw
        
        # 4. Save the new hypothesis and its result to the Cache
        
        cache_collection.add(
            documents=[final_markdown],
            metadatas=[{"hypothesis": target_hypothesis}],
            ids=[str(uuid.uuid4())]
        )
        
        print("\nResearch complete! The final synthesis has been saved to 'research_review.md' and added to the semantic cache.")
        return final_markdown, False

    """
        if len(cache_results['distances'][0])>0 and cache_results['distances'][0][0]<0.25:
    
            #retrieve cache markdown
            cached_markdown = cache_results['documents'][0][0]
    
            with open("research_review.md", "w") as file:
                file.write(cached_markdown)
                
            print("Done! The cached synthesis has been saved to 'research_review.md' (Cost: 0 tokens).")
            
        """




# Use the below code to test code without cache memory system
"""  
# Execute the Engine
if __name__ == "__main__":
    target_hypothesis = "The impact of Retrieval-Augmented Generation (RAG) on reducing hallucination rates in medical diagnostics."
    
    print(f"Initiating research on: {target_hypothesis}")
    
    # The kickoff method starts the sequential execution
    result = research_crew.kickoff(inputs={"hypothesis": target_hypothesis})
    
    print("\nResearch complete! The final synthesis has been saved to 'research_review.md'.")"""