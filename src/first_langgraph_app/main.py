import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END, START
from langchain_google_genai import ChatGoogleGenerativeAI

# Set your API key
os.environ["GOOGLE_API_KEY"] = "Your-KEY-HERE"

# Initialize LLM (using Gemini Flash for speed and cost-efficiency)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

class BlogState(TypedDict):
    topic: str
    research_notes: str
    draft_content: str
    final_content: str
    quality_feedback: str
    quality_score: int

def research_node(state: BlogState) -> BlogState:
    """Research agent gathers information about the topic."""
    topic = state["topic"]

    prompt = f"""You are a research specialist. Gather key information about: {topic}

    Provide 3-5 key points that would be valuable for a blog post.
    Focus on current trends, benefits, and real-world applications.
    """

    response = llm.invoke(prompt)

    return {
        "research_notes": response.content
    }

def write_node(state: BlogState) -> BlogState:
    """Writing agent creates the first draft."""
    topic = state["topic"]
    research = state["research_notes"]

    prompt = f"""You are a content writer. Create a blog post draft about: {topic}

    Research notes:
    {research}

    Write an engaging 300-word blog post with:
    - Catchy introduction
    - Key points from research
    - Practical examples
    - Strong conclusion
    """

    response = llm.invoke(prompt)

    return {
        "draft_content": response.content
    }


def edit_node(state: BlogState) -> BlogState:
    """Editing agent refines the content."""
    draft = state["draft_content"]

    prompt = f"""You are a professional editor. Refine this blog post draft:

    {draft}

    Improve:
    - Clarity and flow
    - Grammar and style
    - Engagement and readability

    Return the polished version.
    """

    response = llm.invoke(prompt)

    return {
        "final_content": response.content
    }

def quality_check_node(state: BlogState) -> BlogState:
    """Quality check agent reviews the final content for accuracy and coherence."""
    final_content = state["final_content"]

    prompt = f"""You are a quality assurance specialist. Review the following blog post for accuracy, coherence, and overall quality:
    
    {final_content}
    
    Provide a score from 1-10 (10 being the best) and feedback for improvements.
    Return a JSON object with two keys: 'score' and 'feedback'.
    """

    response = llm.invoke(prompt)
    response_json = json.loads(response.content.strip('`json\n'))
    
    return {
        "quality_feedback": response_json["feedback"],
        "quality_score": response_json["score"]
    }


def should_revise(state: BlogState) -> str:
    """Determines whether to revise the draft or end the workflow."""
    print("---ASSESSING QUALITY---")
    score = state.get("quality_score", 10)
    if score < 7:
        return "write"  # Revise
    else:
        return END  # Done


def create_blog_workflow():
    # Create graph
    workflow = StateGraph(BlogState)

    # Add nodes
    workflow.add_node("research", research_node)
    workflow.add_node("write", write_node)
    workflow.add_node("edit", edit_node)
    workflow.add_node("quality_check", quality_check_node)

    # Add edges (define the flow)
    workflow.add_edge(START, "research")
    workflow.add_edge("research", "write")
    workflow.add_edge("write", "edit")
    workflow.add_edge("edit", "quality_check")
    #workflow.add_edge("quality_check", END)
    
    workflow.add_conditional_edges(
        "quality_check",
        should_revise
    )

    # Compile
    return workflow.compile()

# Create the app
app = create_blog_workflow()

# Run the workflow
#result = app.invoke({
#    "topic": "Benefits of cloud computing for small businesses"
#})

# View results
#print("=== RESEARCH NOTES ===")
#print(result["research_notes"])
#print("\n=== DRAFT CONTENT ===")
#print(result["draft_content"])
#print("\n=== After Edition ===")
#print(result["final_content"])
#print("\n=== Quality Check Eval ===")
#print(result["quality_feedback"])
# Stream the workflow execution
for chunk in app.stream({"topic": "Benefits of cloud computing for small businesses"}):
    os.system('cls' if os.name == 'nt' else 'clear')
    for key, value in chunk.items():
        print(f"--- {key.upper()} ---")
        print(value)
        print("\n" + "="*30 + "\n")