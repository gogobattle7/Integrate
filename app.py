from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import requests

app = Flask(__name__)

# 전역 변수로 모델을 초기화하지 않음
pipe1 = None
pipe3 = None

# 모델 1: Stable LM 2 1.6B
def load_model1():
    global pipe1
    if pipe1 is None:
        model1_name = "stabilityai/stablelm-2-1_6b"
        pipe1 = pipeline("text-generation", model=model1_name)
    return pipe1

def generate_text_model1(prompt, max_new_tokens=50, temperature=0.7, do_sample=True, top_p=0.9):
    pipe1 = load_model1()
    generated = pipe1(
        prompt,
        max_length=len(prompt.split()) + max_new_tokens,
        num_return_sequences=1,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample
    )
    return generated[0]['generated_text']

# 모델 2: Phi-3-mini-4k-instruct (Hugging Face API)
api_url = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"
headers = {"Authorization": "Bearer ?????"}  # 여기에 API 키를 넣으세요.

def generate_text_model2(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.5,
            "do_sample": False
        }
    }
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

# 모델 3: Flan-T5-xl
def load_model3():
    global pipe3
    if pipe3 is None:
        model3_name = "google/flan-t5-xl"
        tokenizer3 = AutoTokenizer.from_pretrained(model3_name)
        model3 = AutoModelForSeq2SeqLM.from_pretrained(model3_name)
        pipe3 = pipeline("text2text-generation", model=model3, tokenizer=tokenizer3)
    return pipe3

def generate_text_model3(prompt, max_new_tokens=50, temperature=0.5, top_p=0.8, do_sample=True):
    pipe3 = load_model3()
    generated = pipe3(
        prompt,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample
    )
    return generated[0]['generated_text']

# 라우트 설정
@app.route('/')
def main():
    return render_template('main.html')

@app.route('/model1', methods=['GET', 'POST'])
def model1():
    query = None
    response = None
    if request.method == 'POST':
        query = request.form['content']
        try:
            response = generate_text_model1(query)
        except Exception as e:
            response = str(e)
    return render_template('index.html', query=query, response=response)

@app.route('/model2', methods=['GET', 'POST'])
def model2():
    query = None
    response = None
    if request.method == 'POST':
        query = request.form['content']
        try:
            response = generate_text_model2(query)
        except Exception as e:
            response = str(e)
    return render_template('index.html', query=query, response=response)

@app.route('/model3', methods=['GET', 'POST'])
def model3():
    query = None
    response = None
    if request.method == 'POST':
        query = request.form['content']
        try:
            enhanced_prompt = f"Answer the following question clearly and concisely: {query}"
            response = generate_text_model3(enhanced_prompt)
        except Exception as e:
            response = str(e)
    return render_template('index.html', query=query, response=response)

if __name__ == '__main__':
    app.run(debug=True)