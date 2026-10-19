"""RAG Ingest Command - Ingest data into the RAG system.

This module implements the ingest command for the SupportDesk RAG CLI.
It loads and processes tickets, chunks them, generates embeddings, and stores them.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class RagIngestCommand:
    """Command to ingest data into the RAG system."""

    def __init__(self, verbose: bool = False, debug: bool = False):
        self.verbose = verbose or debug
        self.debug = debug

    def run(self, data_path: str, config_path: Optional[str] = None,
            clean: bool = True, wipe: bool = False) -> Dict[str, Any]:
        """Run the ingest command.
        
        Args:
            data_path: Path to the data file (JSON or CSV)
            config_path: Optional path to config file
            clean: Whether to clean PII from text
            
        Returns:
            Dictionary with ingestion results
        """
        from supportdesk_rag import (
            SupportDeskIngest,
            EmbeddingService,
            EmbeddingPipeline,
            create_vector_store,
            StorePipeline
        )
        
        if self.verbose:
            print(f"[INFO] Loading data from {data_path}...")
        
        # Initialize ingest
        ingest = SupportDeskIngest()
        
        try:
            # Load tickets
            tickets = ingest.load_ticket(data_path)
            if self.verbose:
                print(f"[INFO] Loaded {len(tickets)} tickets")
            
            # Process tickets
            chunks = ingest.process_tickets(data_path, clean=clean)
            if self.verbose:
                print(f"[INFO] Generated {len(chunks)} chunks")
            
            # Initialize embedding service
            if self.verbose:
                print("[INFO] Creating embeddings...")
            
            embed_service = EmbeddingService()
            embed_pipeline = EmbeddingPipeline(embed_service)
            
            # Embed chunks
            embedded_chunks = embed_pipeline.process_chunks(chunks)
            if self.verbose:
                print(f"[INFO] Created {len(embedded_chunks)} embeddings")
            
            # Initialize vector store
            if self.verbose:
                print("[INFO] Storing embeddings in vector database...")
            
            vector_store = create_vector_store()
            
            # Wipe collection if requested
            if wipe:
                print(f"[INFO] Wiping collection '{vector_store.collection_name}'...")
                wipe_result = vector_store.wipe()
                print(f"[INFO] Wipe result: {wipe_result}")
            
            store_pipeline = StorePipeline(vector_store=vector_store)
            
            # Store chunks
            storage_result = store_pipeline.store_chunks(embedded_chunks)
            if self.verbose:
                print(f"[INFO] Storage result: {storage_result}")
            
            output = {
                "data_path": data_path,
                "tickets_loaded": len(tickets),
                "chunks_generated": len(chunks),
                "embeddings_created": len(embedded_chunks),
                "storage_result": storage_result,
                "metrics": {}
            }
            
            return output
            
        except Exception as e:
            if self.debug:
                raise
            return {
                "error": str(e),
                "data_path": data_path,
                "chunks_generated": 0,
                "embeddings_created": 0
            }

    def format_output(self, result: Dict[str, Any], output_format: str = "markdown") -> str:
        """Format the output for display."""
        if output_format == "json":
            return json.dumps(result, indent=2, default=str)
        
        # Markdown format
        lines = []
        lines.append("# Data Ingestion Results")
        lines.append("")
        
        if "error" in result:
            lines.append("## ❌ Error")
            lines.append("")
            lines.append(f"```text\n{result['error']}\n```")
            return "\n".join(lines)
        
        lines.append("## 📊 Ingestion Summary")
        lines.append("")
        lines.append(f"- **Data Source:** {result.get('data_path', 'N/A')}")
        lines.append(f"- **Tickets Loaded:** {result.get('tickets_loaded', 0)}")
        lines.append(f"- **Chunks Generated:** {result.get('chunks_generated', 0)}")
        lines.append(f"- **Embeddings Created:** {result.get('embeddings_created', 0)}")
        lines.append("")
        
        storage_result = result.get('storage_result', {})
        if storage_result:
            lines.append("## 💾 Storage Results")
            lines.append("")
            lines.append(f"- **Stored:** {storage_result.get('stored', 0)}")
            lines.append(f"- **Failed:** {storage_result.get('failed', 0)}")
            lines.append(f"- **Skipped:** {storage_result.get('skipped', 0)}")
            lines.append(f"- **Collection:** {storage_result.get('collection', 'N/A')}")
            lines.append(f"- **Total Points:** {storage_result.get('total_points', 0)}")
            lines.append("")
        
        return "\n".join(lines)


def main():
    """Main entry point for the ingest command."""
    parser = argparse.ArgumentParser(
        description="Ingest data into the RAG system"
    )
    parser.add_argument(
        "--data-path", "-d",
        type=str,
        required=True,
        help="Path to data file (JSON or CSV)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--no-clean",
        action="store_false",
        dest="clean",
        default=True,
        help="Don't clean PII from text"
    )
    parser.add_argument(
        "--wipe",
        action="store_true",
        dest="wipe",
        default=False,
        help="Wipe the collection before ingesting new data"
    )
    parser.add_argument(
        "--no-wipe",
        action="store_false",
        dest="wipe",
        default=False,
        help="Keep existing data and append (default behavior)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output in JSON format"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (implies verbose)"
    )
    
    args = parser.parse_args()
    
    command = RagIngestCommand(
        verbose=args.verbose,
        debug=args.debug
    )
    
    result = command.run(
        data_path=args.data_path,
        config_path=args.config,
        clean=args.clean,
        wipe=args.wipe
    )
    
    output_format = "json" if args.json_output else "markdown"
    output = command.format_output(result, output_format)
    print(output)
    
    if "error" in result:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
