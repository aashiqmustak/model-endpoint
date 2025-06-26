import requests
from langgraph.graph import StateGraph, START, END

# ========== Config ==========
LINKEDIN_ACCESS_TOKEN = "AQXDuMVArSfugaUjpcEsc8aShQPfRhjsaPJq9RVdr-rfdnNap7DC2n1O0EeHXq-m37hiDkF0q3xsaIAuEY_5VnfgYIeKvp_udvAG_09uMvoTwPO9rxbhzGx8cKDtvD9dXr7p6rDuVWQNlI-TY2Wm-fpMvMXcWS21i0m_EpWiMpecfa9k9RUDUHPbYIHv3ZHfhtbFiIvie4xb-foaeyjV7E9nj7kIKtQLQRSevDad2bK0lLPy-L62QCpeQz0FxeU5DFHBoPBzQSoB7O9XKn0zyEWH3FFeKEcYZyfkp6WVPWebtzmFxKg03C787qlz4i-D_iY5zekeT0di9WJr92qnEMHskkLmNQ"
PERSON_URN = "urn:li:person:f-rqHhOZHA"
JOB_JSON_URL = "https://0040-137-97-228-158.ngrok-free.app/llm-output"
# ========== Node: Post Job ==========
def post_job_to_linkedin(state: dict) -> dict:
    print("🚀 [post_job_to_linkedin]")

    job = state.get("job_data", {})
    if not job:
        state["job_result"] = "❌ No job data provided."
        return state

    job_title = job.get("job_title", "Job Opening")
    experience = job.get("experience", "N/A")
    location = job.get("location", "Remote")
    skills = ", ".join(job.get("skills", [])) or "Not specified"

    post_text = (
        f"🚀 New Job Opportunity!\n\n"
        f"📌 Title: {job_title}\n"
        f"🧠 Experience: {experience}\n"
        f"📍 Location: {location}\n"
        f"🛠 Skills: {skills}\n\n"
        "#Hiring #JobOpening #Careers"
    )

    headers = {
        "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0"
    }

    payload = {
        "author": PERSON_URN,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": post_text},
                "shareMediaCategory": "NONE"
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
        }
    }

    try:
        res = requests.post("https://api.linkedin.com/v2/ugcPosts", headers=headers, json=payload)
        if res.status_code == 201:
            post_id = res.headers.get("x-restli-id", "unknown")
            state["job_result"] = f"✅ Posted: https://www.linkedin.com/feed/update/{post_id}"
        else:
            state["job_result"] = f"❌ Failed: {res.status_code} - {res.text}"
    except Exception as e:
        state["job_result"] = f"❌ Exception: {e}"

    return state

# ========== Node: Finalize ==========
def finalize(state: dict) -> dict:
    print("✅ [finalize]")
    print(f"📤 {state.get('job_result')}")
    return state

# ========== Build LangGraph ==========
graph = StateGraph(dict)
graph.add_node("post_job", post_job_to_linkedin)
graph.add_node("finalize", finalize)

graph.add_edge(START, "post_job")
graph.add_edge("post_job", "finalize")
graph.add_edge("finalize", END)

app = graph.compile()

# ========== Run ==========
if __name__ == "__main__":
    try:
        # Step 1: Fetch from endpoint
        input_json = requests.get(JOB_JSON_URL, timeout=5).json()

        # Step 2: Extract only 'entities' from 'output'
        output = input_json.get("output", {})
        entities = output.get("entities", {})

        input_state = {
            "job_data": entities  # Only entities are required for LinkedIn
        }

    except Exception as e:
        input_state = {
            "job_data": {},
            "job_result": f"❌ Fetch error: {e}"
        }

    print("📥 Input to LangGraph:", input_state)

    # Step 3: Run the LangGraph workflow
    result = app.invoke(input_state)

    print("\n🟢 DONE")
    print("📦 Output:", result.get("job_result"))