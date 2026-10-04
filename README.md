# Medical Encyclopedia AI

A simple RAG-based medical encyclopedia chatbot built from the college-provided project.

## What is included

- `app.py` — complete frontend + RAG application in one Python file
- `Medical_book.pdf` — supplied medical encyclopedia
- `requirements.txt` — required Python packages
- `.env.example` — API-key configuration template

## How it works

Medical PDF → text extraction → chunks → Hugging Face embeddings → Pinecone → relevant chunks → Groq → chatbot answer.

## Run locally

1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Create a virtual environment if desired:
   `python -m venv venv`
4. Activate it.
5. Install packages:
   `pip install -r requirements.txt`
6. Copy `.env.example` to `.env`.
7. Put your real `GROQ_API_KEY` and `PINECONE_API_KEY` in `.env`.
8. Start the app:
   `streamlit run app.py`

The browser should open the chatbot automatically.

## First-time Pinecone setup

If the `medical-chatbot` index does not already contain the book, open the app and use **Build / Update Pinecone Index** in the sidebar. This reads the supplied PDF, creates the 384-dimensional Pinecone index when needed, and uploads the document chunks.

Do not press the build button repeatedly unless you intentionally want to upload the document again.

## Important deployment notes

- Never publish `.env` or expose API keys in frontend code.
- Before making the supplied medical book publicly accessible or deploying its full contents, confirm that the college/project has the necessary permission or license for public use. The supplied book contains copyright notices.
- Keep the medical disclaimer visible in the deployed application.
- This is an educational information system, not a diagnostic or treatment system.
