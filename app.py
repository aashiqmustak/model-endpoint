from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
import os
import json
import re
from datetime import datetime
from pyngrok import ngrok

app = Flask(__name__)
os.makedirs("offload_dir", exist_ok=True)

# Model & Adapter paths
base_model_id = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_path = "lora_adapter_fast_full_on"

# Global storage for the latest output
latest_llm_output = {}

# Load model
try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    print("Loading base model...")
    config = PeftConfig.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map="auto",
        offload_folder="offload_dir"
    )

    model.eval()
    model_loaded = True
    print("Model loaded successfully!")

except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None
    model_loaded = False

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
                do_sample=True,              # Enable sampling
                temperature=0.7,             # Now effective
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = full_output.split("Output:")[-1].strip() if "Output:" in full_output else full_output.strip()

        # Attempt to parse JSON
        try:
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return fill_defaults(parsed)
        except:
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

# Flask routes

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

# Main runner
if __name__ == '__main__':
    print("Starting Flask server...")
    print(f"Model loaded: {model_loaded}")
    if model_loaded:
        print(f"Base model: {base_model_id}")
        print(f"Adapter: {adapter_path}")

    port = 5000
    try:
        public_url = ngrok.connect(port)
        print(f" * ngrok tunnel available at: {public_url}")
        print(f" * Test endpoints: {public_url}/test or /slack-input")
    except Exception as e:
        print(f" * ngrok failed: {str(e)}")

    app.run(port=port)
