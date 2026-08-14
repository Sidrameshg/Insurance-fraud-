from pathlib import Path
import json
import re


# =================================================
# PROJECT ROOT
# =================================================

ROOT = Path(__file__).resolve().parents[3]

DOCUMENT_DIR = (
    ROOT
    / "phase4"
    / "rag"
    / "documents"
)

OUTPUT_DIR = (
    ROOT
    / "phase4"
    / "rag"
    / "vectorstore"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "documents.json"
)


# =================================================
# SETTINGS
# =================================================

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md"
}

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


# =================================================
# TEXT CLEANING
# =================================================

def clean_text(text):

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# =================================================
# CHUNKING
# =================================================

def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    if not text:

        return []

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[
            start:end
        ].strip()

        if chunk:

            chunks.append(
                chunk
            )

        if end >= len(text):

            break

        start = (
            end
            - overlap
        )

    return chunks


# =================================================
# LOAD DOCUMENTS
# =================================================

def load_documents():

    documents = []

    if not DOCUMENT_DIR.exists():

        raise FileNotFoundError(
            f"Knowledge directory not found: "
            f"{DOCUMENT_DIR}"
        )

    for path in sorted(
        DOCUMENT_DIR.iterdir()
    ):

        if not path.is_file():

            continue

        if path.suffix.lower() not in (
            SUPPORTED_EXTENSIONS
        ):

            continue

        text = path.read_text(
            encoding="utf-8"
        )

        text = clean_text(
            text
        )

        chunks = chunk_text(
            text
        )

        for index, chunk in enumerate(
            chunks
        ):

            documents.append({

                "document_id":
                    path.stem,

                "source":
                    path.name,

                "chunk_id":
                    index,

                "text":
                    chunk
            })

    return documents


# =================================================
# SAVE
# =================================================

def save_documents(
    documents
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {

        "phase":
            "phase4",

        "component":
            "rag",

        "stage":
            "document_ingestion",

        "document_count":
            len(
                set(
                    item["document_id"]
                    for item in documents
                )
            ),

        "chunk_count":
            len(documents),

        "chunks":
            documents
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )


# =================================================
# MAIN
# =================================================

def main():

    print()
    print("==========================================")
    print("       PHASE 4 RAG DOCUMENT INGESTION")
    print("==========================================")

    documents = load_documents()

    print()
    print(
        "Documents loaded:",
        len(
            set(
                item["document_id"]
                for item in documents
            )
        )
    )

    print(
        "Chunks created:",
        len(documents)
    )

    save_documents(
        documents
    )

    print()
    print(
        "Chunk size:",
        CHUNK_SIZE
    )

    print(
        "Chunk overlap:",
        CHUNK_OVERLAP
    )

    print()
    print(
        "Output:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("==========================================")
    print("       RAG INGESTION COMPLETE")
    print("==========================================")


if __name__ == "__main__":

    main()
