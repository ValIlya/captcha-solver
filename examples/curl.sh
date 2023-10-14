(echo -n '{"data": ["'; base64 examples/c3xavu.png; echo '"]}') |
curl -H "Content-Type: application/json" -d @-  http://127.0.0.1:7860/api/predict

(echo -n '{"data": ["'; base64 examples/c3xavu.png; echo '"]}') |
curl -H "Content-Type: application/json" -X POST -d @- https://d5845856c4b55cac98.gradio.live/api/predict

