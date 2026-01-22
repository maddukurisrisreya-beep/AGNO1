def chunk_by_section(text: str):
    chunks = []
    current = ""

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.isupper():
            if current:
                chunks.append(current.strip())
            current = line
        else:
            current += " " + line

    if current:
        chunks.append(current.strip())

    return chunks
