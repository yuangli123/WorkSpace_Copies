from typing import Tuple


def format_memory_content(content: str, source: str) -> str:
    if not source:
        return content
    return f"{content}\n\n----- ORIGINAL SOURCE -----\n{source}"


def parse_memory_content(document: str) -> Tuple[str, str]:
    separator = "\n\n----- ORIGINAL SOURCE -----\n"
    if separator in document:
        parts = document.split(separator)
        
        return parts[0], separator.join(parts[1:]) 
    else:
        return document, ""
