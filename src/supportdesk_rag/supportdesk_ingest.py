import csv
import json
import re
from typing import List, Dict

import tiktoken
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SupportDeskIngest:
    def __init__(self,
                 chunk_size: int=1000,
                 chunk_overlap: int=150,
                 chunk_type: str = "recursive" # "recursive" or "token"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_type = chunk_type


    # load the tickets from the ticket source as a JSON or CSV file
    def load_ticket(self,
                    data_path: str,
                    format: str = "json"
    ) -> List[Dict]:
        file_format = (format or "").lower()
        lower_path = data_path.lower()
        is_json = lower_path.endswith(".json") or (
            file_format == "json" and not lower_path.endswith((".csv", ".tsv"))
        )

        if is_json:
            with open(data_path, "r", encoding="utf-8") as tickets:
                payload = json.load(tickets)

            if isinstance(payload, dict):
                rows = payload.get("tickets", [])
            else:
                rows = payload

            normalized_rows = []
            for ticket in rows:
                row = dict(ticket)
                user = row.pop("user", {})
                row["user_name"] = user.get("name", "")
                row["user_email"] = user.get("email", "")
                row["user_team"] = user.get("team", "")
                row["user_location"] = user.get("location", "")
                normalized_rows.append(row)
            return normalized_rows

        with open(data_path, "r", newline="", encoding="utf-8") as tickets:
            content = csv.DictReader(tickets)
            return list(content)


    # apply guardrails - Do this for PII's only - will extend later
    def clean_text(self,
                   text: str
    ) -> str:
        pii_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]'),  # SSN (if applicable)
            (r'\b\d{16}\b', '[CARD_REDACTED]'),  # Credit card-like patterns
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]'),
        ]

        for pattern, replacement in pii_patterns:
            text = re.sub(pattern, replacement, text)

        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Create  appropriate chunks
    def chunk_text(self,
                   text: str,
                   strategy: str=""
    ) -> List[str]:
        # Use instance chunk_type if no strategy provided
        if strategy == "":
            strategy = self.chunk_type

        if strategy == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", ",", "", " "]
            )
            chunks = splitter.split_text(text)
            return chunks
        elif strategy == "token":
            # Token-based chunking (requires tiktoken)
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            chunks = []
            for i in range(0, len(tokens), self.chunk_size):
                chunk_tokens = tokens[i:i + self.chunk_size]
                chunks.append(encoding.decode(chunk_tokens))
            return chunks
        elif strategy == "semantic":
            raise NotImplementedError("Semantic chunking requires embedding model")
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")


    # apply a simple deduplication strategy - exact  string deduplication
    def dedupe_chunks(self, chunks: List[str]) -> List[str]:
        seen = set()
        unique_chunks = []
        for chunk in chunks:
            if chunk not in seen:
                unique_chunks.append(chunk)
            else:
                seen.add(chunk)

        return unique_chunks

    #  get the chunk and the metadata
    def process_tickets(self,
                    data_path: str,
                    format: str = "json",
                    clean: bool = True
    ) -> List[Dict]:
        chunks = []
        tickets = self.load_ticket(data_path, format)
        for ticket in tickets:
            # Combine description and resolution_notes for better searchability
            ticket_text = ticket["description"]
            if ticket.get("resolution_notes") and ticket["resolution_notes"].strip():
                ticket_text += "\n\nResolution: " + ticket["resolution_notes"]

            # Clean the text before chunking (removes PII, normalizes whitespace)
            text_to_chunk = self.clean_text(ticket_text) if clean else ticket_text  # clean up the text
            description_chunk = self.chunk_text(text_to_chunk)  # chunk the text
            description_chunk = self.dedupe_chunks(description_chunk) # dedupe the chunks
            for i, chunk_text in enumerate(description_chunk): # attach metadata
                the_chunk = {
                    "ticket_id": ticket["ticket_id"],
                    "chunk_text": chunk_text,
                    "chunk_index": i,
                    "metadata": {
                        "service": ticket["service"],
                        "severity": ticket["severity"],
                        "ticket_type": ticket["ticket_type"],
                        "created_at": ticket["created_at"],
                        "user_name": ticket["user_name"],
                        "user_email": ticket["user_email"]
                    }
                }
                chunks.append(the_chunk)

                print(f"Chunk {i+1}/{len(description_chunk)} for ticket {ticket['ticket_id']}")

                #import pdbpp
                #pdbpp.set_trace()


        return chunks

    def get_chunks(self,
                   data_path: str,
                   format: str = "json",
                   clean: bool = True
    ) -> List[Dict]:
        return self.process_tickets(data_path, format=format, clean=clean)


def main():
    data_path = "./data/synthetic_tickets.json"
    ingest = SupportDeskIngest(chunk_size=100, chunk_overlap=10)
    tickets = ingest.process_tickets(data_path, clean=True)

if __name__ == "__main__":
    main()