import argparse
import logging
import os
from pathlib import Path
from typing import List

from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


class Config:
    DATA_DIR = "data"
    DB_PATH = "latinragdb"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = os.getenv("OLLAMA_MODEL", "mixtral")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    RETRIEVER_K = 4
    MAX_RETRIES = 3
    TRANSLATION_TIMEOUT = 120


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def validate_file_path(path: str) -> bool:
    path_obj = Path(path)
    if not path_obj.exists():
        logger.error("File not found: %s", path)
        return False
    if not os.access(path, os.R_OK):
        logger.error("File not readable: %s", path)
        return False
    return True


def create_document(text: str, metadata: dict) -> Document:
    return Document(page_content=text, metadata=metadata)


def load_parallel_texts() -> List[Document]:
    docs: List[Document] = []
    if not os.path.isdir(Config.DATA_DIR):
        return docs

    latin_files = [f for f in os.listdir(Config.DATA_DIR) if f.endswith("_latin.txt")]

    for latin_file in latin_files:
        english_file = latin_file.replace("_latin.txt", "_english.txt")
        english_path = os.path.join(Config.DATA_DIR, english_file)
        latin_path = os.path.join(Config.DATA_DIR, latin_file)

        if not validate_file_path(english_path) or not validate_file_path(latin_path):
            continue

        try:
            with open(latin_path, "r", encoding="utf-8") as lat_file, open(
                english_path, "r", encoding="utf-8"
            ) as eng_file:
                latin_lines = lat_file.readlines()
                english_lines = eng_file.readlines()

                current_chapter = ""
                current_text = ""

                for lat, eng in zip(latin_lines, english_lines):
                    lat = lat.strip()
                    eng = eng.strip()

                    if lat.startswith("CAPUT"):
                        if current_text:
                            docs.append(
                                create_document(
                                    current_text,
                                    {
                                        "chapter": current_chapter,
                                        "type": "parallel_text",
                                    },
                                )
                            )
                        current_chapter = lat
                        current_text = ""
                    elif lat and eng:
                        current_text += f"Latin: {lat}\nEnglish: {eng}\n\n"

                if current_text:
                    docs.append(
                        create_document(
                            current_text,
                            {"chapter": current_chapter, "type": "parallel_text"},
                        )
                    )
        except Exception as exc:
            logger.error("Failed to process %s: %s", latin_file, exc)

    return docs


def load_documents() -> List[Document]:
    docs = load_parallel_texts()
    if not os.path.isdir(Config.DATA_DIR):
        return docs

    for filename in os.listdir(Config.DATA_DIR):
        path = os.path.join(Config.DATA_DIR, filename)
        lower = filename.lower()

        if lower.endswith("_latin.txt") or lower.endswith("_english.txt"):
            continue
        if not validate_file_path(path):
            continue

        try:
            if lower.endswith(".txt"):
                logger.info("Loading TXT: %s", filename)
                loader = TextLoader(path, encoding="utf-8")
                docs.extend(loader.load())
            elif lower.endswith(".pdf"):
                logger.info("Loading PDF: %s", filename)
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
        except Exception as exc:
            logger.error("Failed to load %s: %s", filename, exc)

    return docs


def initialize_llm() -> Ollama:
    try:
        logger.info("Initializing Ollama client model=%s base_url=%s", Config.LLM_MODEL, Config.OLLAMA_BASE_URL)
        llm = Ollama(
            model=Config.LLM_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.3,
            timeout=Config.TRANSLATION_TIMEOUT,
        )
        llm("test")
        return llm
    except Exception as exc:
        logger.error("LLM initialization failed: %s", exc)
        raise


def invoke_translator(engine, prompt: str) -> str:
    if isinstance(engine, RetrievalQA):
        result = engine({"query": prompt})
        return str(result.get("result", "")).strip()

    response = engine(prompt)
    if isinstance(response, dict) and "result" in response:
        return str(response["result"]).strip()
    return str(response).strip()


def translate_segment(engine, text: str, direction: str) -> str:
    source_lang = "Latin" if direction == "lat-en" else "English"
    target_lang = "English" if direction == "lat-en" else "Latin"
    prompt = (
        f"[INST] Translate this {source_lang} to {target_lang} exactly as written:\n{text}\n\n"
        "Requirements:\n"
        "- Preserve all numbers and formatting\n"
        "- Maintain original structure\n"
        "- Never add explanations\n"
        "[/INST]"
    )

    for attempt in range(Config.MAX_RETRIES):
        try:
            return invoke_translator(engine, prompt)
        except Exception:
            if attempt == Config.MAX_RETRIES - 1:
                raise
            logger.warning("Attempt %s failed, retrying...", attempt + 1)

    return ""


def translate_file(engine, input_file: str, direction: str = "lat-en") -> str:
    output_file = input_file.replace(".txt", "_translated.txt")
    logger.info("Translating %s to %s", input_file, output_file)

    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:
        for raw_line in infile.readlines():
            line = raw_line.strip()
            if not line:
                continue

            handled = False
            source_lang = "Latin" if direction == "lat-en" else "English"
            target_lang = "English" if direction == "lat-en" else "Latin"

            if line.isupper() and line.startswith("LIBER"):
                result = invoke_translator(
                    engine,
                    f"Translate this {source_lang} book title to {target_lang}: {line}",
                )
                outfile.write(f"{result}\n\n")
                handled = True

            if not handled and line[0].isupper() and not line[0].isdigit():
                result = invoke_translator(
                    engine,
                    f"Translate this {source_lang} section title to {target_lang}: {line}",
                )
                outfile.write(f"{result}\n\n")
                handled = True

            if not handled and line[0].isdigit():
                period_positions = [idx for idx, ch in enumerate(line) if ch == "."]
                if len(period_positions) >= 2:
                    section_nums = line[: period_positions[1] + 1].strip()
                    content = line[period_positions[1] + 1 :].strip()
                    result = invoke_translator(
                        engine,
                        f"Translate this {source_lang} text to {target_lang}: {content}",
                    )
                    outfile.write(f"{section_nums}: {result}\n\n")
                    handled = True

            if not handled:
                result = translate_segment(engine, line, direction)
                outfile.write(f"{result}\n\n")

    logger.info("Translation completed")
    return output_file


def build_qa_chain(llm: Ollama) -> RetrievalQA:
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    db = Chroma.from_documents(chunks, embedding, persist_directory=Config.DB_PATH)
    db.persist()

    retriever = db.as_retriever(search_kwargs={"k": Config.RETRIEVER_K})
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Latin RAG Translator")
    parser.add_argument("--file", "-f", help="Input file to translate")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--phrase", "-p", help="Single phrase to translate")
    parser.add_argument(
        "--direction",
        "-d",
        choices=["lat-en", "en-lat"],
        default="lat-en",
        help="Translation direction: Latin to English (lat-en) or English to Latin (en-lat)",
    )
    parser.add_argument(
        "--use-rag",
        "-r",
        action="store_true",
        help="Use RAG context for file translation (default is direct LLM)",
    )
    args = parser.parse_args()

    llm = initialize_llm()

    if args.phrase:
        result = translate_segment(llm, args.phrase, args.direction)
        print(f"\nTranslation: {result}")
        return

    qa_chain = None
    if args.use_rag or args.interactive:
        print("Loading and preparing documents...")
        qa_chain = build_qa_chain(llm)

    if args.file:
        if args.use_rag and qa_chain is not None:
            print("Using RAG-based translation...")
            translate_file(qa_chain, args.file, direction=args.direction)
        else:
            print("Using direct LLM translation...")
            translate_file(llm, args.file, direction=args.direction)
        return

    if args.interactive and qa_chain is not None:
        prompt_lang = "Latin" if args.direction == "lat-en" else "English"
        source_lang = "Latin" if args.direction == "lat-en" else "English"
        target_lang = "English" if args.direction == "lat-en" else "Latin"

        while True:
            query = input(f"\nEnter {prompt_lang} phrase (or 'exit' to quit): ")
            if query.strip().lower() == "exit":
                break

            result = qa_chain(
                {
                    "query": f"Translate and explain this {source_lang} text to {target_lang}: {query}"
                }
            )
            print("\nTranslation & Explanation:\n", result["result"])

            print("\nSources used:")
            for doc in result.get("source_documents", [])[:2]:
                print("-" * 40)
                print(doc.page_content[:200] + "...")
        return

    print("Please specify either --file, --interactive, or --phrase")


if __name__ == "__main__":
    main()
