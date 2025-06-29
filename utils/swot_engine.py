import re
import openai

# Set OpenAI client externally via dependency injection in app.py
openai_client = None

def set_openai_client(client):
    """
    Sets the global OpenAI client for later use.
    This allows API key to be stored securely in app.py.
    """
    global openai_client
    openai_client = client


def generate_swot_from_text(text):
    """
    Generates a SWOT analysis using the OpenAI API based on provided review text.
    Returns a dictionary with strengths, weaknesses, opportunities, and threats.
    """
    if not openai_client:
        raise ValueError("OpenAI client is not set. Call set_openai_client(client) first.")

    prompt = (
        "You are a business analyst. Based on the following customer feedback text, "
        "generate a concise SWOT analysis. Focus on recurring themes and insights:\n\n"
        f"{text}\n\n"
        "Respond in this structured format:\n"
        "Strengths:\n- ...\nWeaknesses:\n- ...\nOpportunities:\n- ...\nThreats:\n- ..."
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are an expert SWOT analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=600
        )

        output = response.choices[0].message.content
        return parse_swot_response(output)

    except Exception as e:
        return {"error": str(e)}


def parse_swot_response(text):
    """
    Parses structured SWOT text into a dictionary.
    """
    sections = {
        "Strengths": [],
        "Weaknesses": [],
        "Opportunities": [],
        "Threats": []
    }

    current = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Strengths"):
            current = "Strengths"
        elif line.startswith("Weaknesses"):
            current = "Weaknesses"
        elif line.startswith("Opportunities"):
            current = "Opportunities"
        elif line.startswith("Threats"):
            current = "Threats"
        elif line.startswith("-") and current:
            sections[current].append(line[1:].strip())

    return sections
