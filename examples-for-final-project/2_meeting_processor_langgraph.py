from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict, List
import os
from tkinter import Tk, filedialog
from groq import Groq

# Configuration
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

# State definition
class State(TypedDict):
    notes: str
    participants: List[str]
    topics: List[str]
    action_items: List[str]
    minutes: str
    summary: str

# ============= WORKFLOW NODES =============

def extract_participants(state: State) -> State:
    """Extracts the meeting participants."""
    prompt = f"""
    From the following meeting notes, extract ONLY the names of the participants.

    Notes: {state['notes']}

    Respond ONLY with a list of names separated by commas, with no additional explanations.
    Example: John Smith, Mary Johnson, Carl Brown
    """

    response = llm.invoke(prompt)
    participants = [p.strip() for p in response.content.split(',') if p.strip()]

    print(f"✓ Participants extracted: {len(participants)} people")

    return {
        'participants': participants
    }

def identify_topics(state: State) -> State:
    """Identifies the main topics discussed."""
    prompt = f"""
    Identify the 3-5 main topics discussed in this meeting.

    Notes: {state['notes']}

    Respond ONLY with the topics separated by semicolons (;).
    Example: System architecture; Delivery deadlines; Task assignment
    """

    response = llm.invoke(prompt)
    topics = [t.strip() for t in response.content.split(';') if t.strip()]

    print(f"✓ Topics identified: {len(topics)} topics")

    return {
        'topics': topics
    }

def extract_actions(state: State) -> State:
    """Extracts the agreed-upon actions and their owners."""
    prompt = f"""
    Extract the specific actions agreed upon in the meeting, including the owner if mentioned.

    Notes: {state['notes']}

    Response format: one action per line, separated by |
    Example: Mary will handle the backend | Carl will prepare the testing plan | Next meeting on Monday

    If there are no clear actions, respond with: "No specific actions identified"
    """

    response = llm.invoke(prompt)

    if "No specific actions" in response.content:
        action_items = []
    else:
        action_items = [a.strip() for a in response.content.split('|') if a.strip()]

    print(f"✓ Actions extracted: {len(action_items)} items")

    return {
        'action_items': action_items
    }

def generate_minutes(state: State) -> State:
    """Generates formal meeting minutes."""
    participants_str = ", ".join(state['participants'])
    topics_str = "\n• ".join(state['topics'])
    actions_str = "\n• ".join(state['action_items']) if state['action_items'] else "No specific actions were defined"

    prompt = f"""
    Generate formal and professional meeting minutes based on the following information:

    PARTICIPANTS: {participants_str}

    TOPICS DISCUSSED:
    • {topics_str}

    AGREED ACTIONS:
    • {actions_str}

    ORIGINAL NOTES: {state['notes']}

    Generate professional minutes of at most 150 words that include:
    1. Header with meeting type
    2. List of attendees
    3. Main points discussed
    4. Agreements and next steps

    Use a formal tone and a clear structure.
    """

    response = llm.invoke(prompt)

    print(f"✓ Minutes generated: {len(response.content.split())} words")

    return {
        'minutes': response.content
    }

def create_summary(state: State) -> State:
    """Creates an ultra-brief executive summary."""
    prompt = f"""
    Create an executive summary of AT MOST 2 lines (30 words) that captures the essence of this meeting.

    Participants: {', '.join(state['participants'][:3])}{'...' if len(state['participants']) > 3 else ''}
    Main topic: {state['topics'][0] if state['topics'] else 'General'}
    Key actions: {len(state['action_items'])} actions defined

    The summary should be concise and to the point.
    """

    response = llm.invoke(prompt)

    print(f"✓ Summary created")

    return {
        'summary': response.content
    }

# ============= GRAPH CONSTRUCTION =============

def create_workflow():
    """Creates and configures the LangGraph workflow."""
    workflow = StateGraph(State)

    # Add all nodes
    workflow.add_node("extract_participants", extract_participants)
    workflow.add_node("identify_topics", identify_topics)
    workflow.add_node("extract_actions", extract_actions)
    workflow.add_node("generate_minutes", generate_minutes)
    workflow.add_node("create_summary", create_summary)

    # Configure sequential flow
    workflow.add_edge(START, "extract_participants")
    workflow.add_edge("extract_participants", "identify_topics")
    workflow.add_edge("identify_topics", "extract_actions")
    workflow.add_edge("extract_actions", "generate_minutes")
    workflow.add_edge("generate_minutes", "create_summary")
    workflow.add_edge("create_summary", END)

    return workflow.compile()

# ============= PROCESSING FUNCTIONS =============

def transcribe_media_direct(file_path: str) -> str:
    """Transcribes audio/video using the Groq Whisper API directly."""
    try:
        print("🎙️ Transcribing with Groq Whisper API...")

        client = Groq()  # Uses GROQ_API_KEY from the environment

        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                prompt="This is a work meeting with multiple participants.",
                response_format="text"
            )

        print(f"✓ Transcription complete: {len(transcript)} characters")
        return transcript

    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return f"Error: {str(e)}"

def process_meeting_notes(notes: str, app):
    """Processes a single meeting note."""
    initial_state = {
        'notes': notes,
        'participants': [],
        'topics': [],
        'action_items': [],
        'minutes': '',
        'summary': ''
    }

    print("\n" + "="*60)
    print("🔄 Processing meeting note...")
    print("="*60)

    result = app.invoke(initial_state)
    return result

def display_results(result: State, meeting_num: int):
    """Displays the results in a structured way."""
    print(f"\n📋 RESULTS - MEETING #{meeting_num}")
    print("-"*60)

    print(f"\n👥 Participants ({len(result['participants'])}):")
    for p in result['participants']:
        print(f"   • {p}")

    print(f"\n📍 Topics covered ({len(result['topics'])}):")
    for t in result['topics']:
        print(f"   • {t}")

    print(f"\n✅ Agreed actions ({len(result['action_items'])}):")
    if result['action_items']:
        for a in result['action_items']:
            print(f"   • {a}")
    else:
        print("   • No specific actions were defined")

    print(f"\n📄 FORMAL MINUTES:")
    print("-"*40)
    print(result['minutes'])
    print("-"*40)

    print(f"\n💡 EXECUTIVE SUMMARY:")
    print(f"   {result['summary']}")

    print("\n" + "="*60)

# ============= DEMO =============

if __name__ == "__main__":
    app = create_workflow()

    # Small GUI: file picker
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a video or transcript",
        filetypes=[
            ("Video/Audio", "*.mp4 *.mov *.m4a *.mp3 *.wav *.mkv *.webm"),
            ("Text", "*.txt *.md")
        ]
    )

    if not file_path:
        print("No file selected.")
        raise SystemExit(0)

    ext = os.path.splitext(file_path)[1].lower()
    media_exts = {".mp4", ".mov", ".m4a", ".mp3", ".wav", ".mkv", ".webm"}

    if ext in media_exts:
        notes = transcribe_media_direct(file_path)
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            notes = f.read()

    result = process_meeting_notes(notes, app)
    display_results(result, 1)
