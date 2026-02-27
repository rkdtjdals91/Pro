import os
import tempfile
import json
import time
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app, origins="*", allow_headers=["Content-Type", "X-API-Key"], methods=["GET", "POST", "OPTIONS"])

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,X-API-Key')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

@app.route('/')
def index():
    return '🎬 VisionIQ Server Running'

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return '', 204

    api_key = request.form.get('api_key')
    if not api_key:
        return jsonify({'error': 'API 키가 없습니다.'}), 400

    genai.configure(api_key=api_key)

    video_file = request.files.get('video')
    video_url = request.form.get('url')
    tmp_path = None

    try:
        if video_file:
            suffix = os.path.splitext(video_file.filename)[1] or '.mp4'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                video_file.save(tmp.name)
                tmp_path = tmp.name
        elif video_url:
            r = requests.get(video_url, stream=True, timeout=60)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                for chunk in r.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                tmp_path = tmp.name
        else:
            return jsonify({'error': '영상 파일 또는 URL이 필요합니다.'}), 400

        print(f"Uploading to Gemini: {tmp_path}")
        uploaded = genai.upload_file(tmp_path)

        while uploaded.state.name == 'PROCESSING':
            time.sleep(3)
            uploaded = genai.get_file(uploaded.name)

        if uploaded.state.name == 'FAILED':
            return jsonify({'error': '영상 처리 실패'}), 500

        model = genai.GenerativeModel('gemini-2.0-flash')

        prompt = """이 영상을 완전히 분석해주세요. 소리와 화면을 모두 보고 들어서 분석하세요.

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만:

{
  "transcript": [
    {"time": "00:00", "speaker": "A", "text": "실제 대사 내용"},
    {"time": "00:15", "speaker": "B", "text": "실제 대사 내용"}
  ],
  "visual": [
    {"time": "00:00", "desc": "실제 화면 설명", "tags": ["태그1", "태그2"]},
    {"time": "00:30", "desc": "실제 화면 설명", "tags": ["태그1"]}
  ],
  "report": {
    "duration": "분:초",
    "speakers": 2,
    "scenes": 4,
    "summary": "영상 전체 요약 3-4문장",
    "highlights": ["핵심 포인트1", "핵심 포인트2", "핵심 포인트3"]
  }
}"""

        response = model.generate_content([uploaded, prompt])
        text = response.text.strip()

        if '```' in text:
            text = text.split('```')[1]
            if text.startswith('json'):
                text = text[4:]

        result = json.loads(text.strip())
        return jsonify({'success': True, 'data': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
