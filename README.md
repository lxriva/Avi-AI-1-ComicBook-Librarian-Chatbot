# 🤖 Comic Book Librarian Chatbot

A local AI-powered chatbot that answers detailed questions about the **DC and Marvel comic universes**, built using [ComicVine](https://comicvine.gamespot.com/api/) volume data and embedded with [LangChain](https://www.langchain.com/), [FAISS](https://github.com/facebookresearch/faiss), and [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings).

> **Ask questions like:**  
> "What happens in *Crisis on Infinite Earths*?"  
> "When did *Secret Wars* come out?"  
> "List major Batman comic arcs."

---

## 📚 Features

- ✅ Answers queries about **Marvel and DC comic books**
- 🔍 Searches over **10,000 comic volumes** locally
- ⚡ Fast retrieval using **FAISS vector search**
- 🧠 RAG pipeline built with **LangChain + OpenAI**
- 🌐 Modern HTML/CSS chatbot UI (`chat.html`)
- ⌨️ Submit messages using **Enter** key or **Send** button
- 🧰 Comic data is **manually updated monthly** via `update_comics.py`

---

## 📁 Project Structure

| File / Folder                    | Purpose                                           |
|----------------------------------|---------------------------------------------------|
| `chat.html`                      | Chatbot front-end UI (styled, mobile-friendly)   |
| `chatbot_server.py`             | Flask server handling question/answer queries     |
| `parse_comicvine_data.py`       | Parses ComicVine volumes into clean JSON chunks   |
| `embed_comicvine_data.py`       | Embeds data into FAISS vector index               |
| `update_comics.py`              | Fetches & saves new monthly DC/Marvel volumes     |
| `fetch_comicvine_volumes.py`    | Fetches raw data from ComicVine API               |
| `comic_corpus.json`             | Cleaned and filtered comic data (DC & Marvel)     |
| `comicvine_index/`              | FAISS vector index directory                      |
| `updates/`                      | JSON snapshots of monthly updates                 |

---

## 🚀 Quick Start Guide

### 🔧 1. Install Requirements

Ensure you're using **Python 3.10+**, then install dependencies:

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, create one with these dependencies:

```txt
flask
flask-cors
openai
langchain
langchain-community
langchain-openai
faiss-cpu
tiktoken
python-dotenv
```

### 🔑 2. Set Your OpenAI API Key

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 📦 3. Generate the Comic Index (First Time Only)

```bash
python parse_comicvine_data.py
python embed_comicvine_data.py
```

This will:
- Fetch and parse ~10,000 comic volumes from ComicVine
- Embed the content using OpenAI Embeddings
- Store everything in a local FAISS vector database

### ▶️ 4. Start the Server

```bash
python chatbot_server.py
```

This runs a Flask server on: `http://localhost:5000`

### 💬 5. Open the Chat Interface

Open `chat.html` in your browser (directly or via Live Server):
1. Type in your comic question
2. Press **Enter** or click **Send**
3. The AI will respond with detailed comic knowledge

---

## 🆕 Updating Monthly Comic Data

To fetch and embed new DC & Marvel volumes (e.g., for new releases):

```bash
python update_comics.py
python embed_comicvine_data.py
```

This process:
- Filters only DC Comics and Marvel content
- Saves JSON snapshot in `/updates/` directory
- Regenerates the FAISS index with new data

---

## 🎨 Customization

### Adding Comic Background

You can add a comic-style background in `chat.html`:

```css
body {
  background-image: url('background.jpg');
  background-size: cover;
  background-position: center;
}
```

### Modifying UI Styles

The chatbot interface is fully customizable through the CSS in `chat.html`. You can modify colors, fonts, and layout to match your preferred comic book aesthetic.

---

## 🛠️ Technical Details

### Architecture

The chatbot uses a **Retrieval-Augmented Generation (RAG)** architecture:

1. **Data Ingestion**: Comic data is fetched from ComicVine API
2. **Processing**: Raw data is cleaned and chunked for optimal retrieval
3. **Embedding**: Text chunks are embedded using OpenAI's embedding models
4. **Storage**: Embeddings are stored in a FAISS vector index for fast similarity search
5. **Retrieval**: User queries are embedded and matched against the comic corpus
6. **Generation**: Retrieved context is used to generate accurate, detailed responses

### Data Sources

- **ComicVine API**: Primary source for comic book metadata and descriptions
- **DC Comics & Marvel Comics**: Focused exclusively on these two major publishers
- **Monthly Updates**: Regular data refreshes to include new releases

---

## 🤝 Contributing

Contributions are welcome! Here are some ways you can help:

- **Bug Reports**: Found an issue? Please open a GitHub issue
- **Feature Requests**: Have ideas for new features? Let us know!
- **Data Improvements**: Help improve comic data quality and coverage
- **UI Enhancements**: Make the chatbot interface even better

---

## 📜 License & Legal

This project is for **educational and personal use only**.

- All comic content belongs to their respective publishers: **DC Comics** and **Marvel Comics**
- ComicVine data is used under their API terms of service
- OpenAI embeddings are used according to OpenAI's usage policies

---

## 🦸‍♂️ About

**Created by Avi** — for comic fans who wish their librarian was a superhero.

Whether you're a longtime comic reader, a newcomer to the medium, or a researcher studying comic book history, this chatbot is designed to be your knowledgeable companion in exploring the rich universes of DC and Marvel Comics.

---

## 🚨 Troubleshooting

### Common Issues

**Q: The chatbot isn't responding to my questions**
- Check that your OpenAI API key is correctly set in the `.env` file
- Ensure the Flask server is running on `localhost:5000`
- Verify that the FAISS index was created successfully

**Q: I'm getting API rate limit errors**
- OpenAI has rate limits on their API. Wait a moment and try again
- Consider upgrading your OpenAI plan for higher rate limits

**Q: The comic data seems outdated**
- Run the monthly update process: `python update_comics.py && python embed_comicvine_data.py`

**Q: Installation issues with FAISS**
- Try installing `faiss-gpu` instead of `faiss-cpu` if you have a compatible GPU
- On some systems, you may need to install additional dependencies

---

## 📊 Statistics

- **~10,000** comic volumes indexed
- **DC Comics & Marvel Comics** exclusive focus
- **Monthly updates** for fresh content
- **Vector search** for accurate retrieval
- **Mobile-friendly** responsive design

Ready to explore the comic multiverse? Fire up the chatbot and start asking questions! 🚀