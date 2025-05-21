# Chatbot Theme Identifier

This project is a Streamlit-based Document Research & Theme Identification Chatbot
using ChromaDB, Sentence Transformers, and OCR for scanned PDFs.

## Setup

pip install -r backend/requirements.txt

## Folder structure:

chatbot_theme_identifier/
├── backend/
│ ├── app/
│ │ ├── api/
│ │ ├── core/
│ │ ├── models/
│ │ ├── services/
│ │ ├── main.py
│ │ └── config.py
│ ├── data/
│ ├── Dockerfile
│ └── requirements.txt
├── docs/
├── tests/
├── demo/
└── README.md

## Setup and Installation

1. Clone this repository:

git clone <your_repo_url>
cd chatbot_theme_identifier/backend

2. Create and activate a Python virtual environment:

python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

3. To start the Streamlit application, run the following command from the root directory:

streamlit run backend/app/main.py

This will launch the web app locally, usually accessible at http://localhost:8501 in your browser.

Usage:

Upload PDFs or scanned PDFs from the sidebar.
Ask questions to query across uploaded documents.
View document excerpts and synthesized themes.

Notes
Make sure Tesseract OCR is installed on your system for scanned PDF processing.
For any issues or bugs, please open an issue in the repository.