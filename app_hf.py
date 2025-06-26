from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
import re
from datetime import datetime
from pyngrok import ngrok

# Initialize Flask app
app = Flask(__name__)

# Load model and tokenizer
base_model_id = "aashiqmustak/intent-entity"
adapter_path = None
model_loaded = False

try:
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model.eval()
    model_loaded = True
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    model = None
    tokenizer = None

# Global variable for storing the last LLM output
latest_llm_output = {}

# Expected response schema
expected_structure = {
    "intent": None,
    "entities": {
        "job_title": None,
        "experience": None,
        "expiration_date": None,
        "job_type": None,
        "number_of_people_to_hire": None,
        "skills": [],
        "location": None
    }
}

def fill_defaults(parsed_json):
    filled = {
        "intent": parsed_json.get("intent", None),
        "entities": expected_structure["entities"].copy()
    }
    if isinstance(parsed_json.get("entities"), dict):
        for key in expected_structure["entities"]:
            filled["entities"][key] = parsed_json["entities"].get(key, expected_structure["entities"][key])
    return filled

def generate_response(instruction: str, input_text: str = "", max_new_tokens=150):
    if not model_loaded:
        return {"error": "Model not loaded properly"}

    try:
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = full_output.split("Output:")[-1].strip() if "Output:" in full_output else full_output.strip()

        # Try to extract JSON
        try:
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return fill_defaults(parsed)
        except Exception:
            pass

        return {"extracted_text": result, "raw_output": result}

    except Exception as e:
        return {"error": f"Generation failed: {str(e)}"}

def generate_json_response(input_text: str):
    instruction = (
        "Extract the intent and entities from the following user input. "
        "Return the result strictly in JSON format like:\n"
        "{\n"
        "  \"intent\": \"<intent>\",\n"
        "  \"entities\": {\n"
        "    \"key1\": \"value1\",\n"
        "    \"key2\": \"value2\"\n"
        "  }\n"
        "}"
    )
    return generate_response(instruction, input_text)

# Routes
@app.route('/slack-input', methods=['POST'])
def slack_input():
    try:
        data = request.get_json()
        message = data.get("text", "")
        if not message:
            return jsonify({"error": "No message provided"}), 400

        output = generate_json_response(message)

        global latest_llm_output
        latest_llm_output = {
            "input_message": message,
            "output": output,
            "timestamp": datetime.now().isoformat()
        }

        return jsonify({"status": "processed", "output": output})

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/llm-output', methods=['GET'])
def get_llm_output():
    if not latest_llm_output:
        return jsonify({"error": "No output available yet"})
    return jsonify(latest_llm_output)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    input_text = data.get("text", "")
    if not input_text:
        return jsonify({"error": "Input text is required"}), 400

    instruction = "Extract intent and entities"
    result = generate_response(instruction, input_text)
    return jsonify({"output": result})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "running",
        "model_loaded": model_loaded,
        "base_model": base_model_id,
        "adapter_path": adapter_path,
        "device": str(model.device) if model_loaded else "unknown",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    })

@app.route('/test', methods=['GET'])
def test_endpoint():
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 500

    test_message = "I want to hire a Frontend Developer with 4 years of experience"
    result = generate_json_response(test_message)

    return jsonify({
        "test_input": test_message,
        "test_output": result,
        "status": "Model working correctly"
    })

# Start Flask + ngrok
if __name__ == '__main__':
    port = 5000
    public_url = ngrok.connect(port)
    print(f" * ngrok tunnel URL: {public_url}")
    app.run(port=port)
