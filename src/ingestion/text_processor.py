import traceback
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.config import CHUNK_SIZE, CHUNK_OVERLAP

def convert_to_text(json_data: dict) -> str:
    lines = []
    try:
        for key, value in json_data.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                lines.append(f"{key}: {value}")
            elif isinstance(value, (list, tuple)):
                s = f"{key}:\n"
                for item in value:
                    s += f" - {item}\n"
                lines.append(s)
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    lines.append(f"{key}.{sub_key}: {sub_value}")
            else:
                lines.append(f"{key}: {str(value)}")
    except Exception:
        print(f"convert_to_text error: {traceback.format_exc()}")
    return "\n".join(lines)


def split_text(text: str, chunk_size: int = CHUNK_SIZE, 
               chunk_overlap: int = CHUNK_OVERLAP) -> list[Document]:
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators = ["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents([Document(page_content=text)])