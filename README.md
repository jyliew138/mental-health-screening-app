# Text-Based Mental Health Disorder Classification

Streamlit user-facing app for a text-based mental health disorder classification project.

## Target Output Classes

- Anxiety
- Bipolar
- Stress
- Personality Disorder
- Normal
- Depression
- Suicidal

## Current Status

The app interface is ready and the Hugging Face BERT text-classification model is connected.

Model folder path:

```text
best_model/best_model
```

The folder should contain `config.json`, `tokenizer.json`, `tokenizer_config.json`, and `model.safetensors`.

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## User App Structure

1. Home page with the purpose of the screening tool.
2. Personal details page for organizing records.
3. Text screening page for user input.
4. Result page for predicted class and confidence.
5. History page for previous screening records.
6. About page with responsible-use notes and developer model details.
7. Finish page with a closing reminder.

## Important Note

This app is a screening support tool only. It should not be used as a medical diagnosis tool.
