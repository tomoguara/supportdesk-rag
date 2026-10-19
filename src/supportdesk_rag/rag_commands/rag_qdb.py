"""RAG QDB Command - Query Qdrant database directly for debugging.

This module implements the qdb command for the SupportDesk RAG CLI.
It allows direct querying of the vector database for debugging purposes.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class RagQdbCommand:
    """Command to query Qdrant database directly."""

    def __init__(self, verbose: bool = False, debug: bool = False):
        self.verbose = verbose or debug
        self.debug = debug

    def run(
        self,
        info: bool = False,
        list: Optional[int] = None,
        ticket_id: Optional[str] = None,
        search: Optional[str] = None,
        filter: bool = False,
        service: Optional[str] = None,
        severity: Optional[str] = None,
        top_k: int = 5,
        limit: int = 20
    ) -> Dict[str, Any]:
        """Run the qdb command.

        Args:
            info: Show collection information
            list: List all ticket IDs (number of tickets to show, or True for default 20)
            ticket_id: Search for specific ticket ID
            search: Search by text similarity
            filter: Filter by metadata
            service: Filter by service (use with filter)
            severity: Filter by severity (use with filter)
            top_k: Number of results for search
            limit: Limit for list command (deprecated, use list parameter instead)

        Returns:
            Dictionary with query results
        """
        from supportdesk_rag.supportdesk_store import VectorStore

        try:
            # Initialize vector store
            store = VectorStore()

            # Check health
            if not store.health_check():
                return {
                    "error": "Cannot connect to Qdrant. Is it running?",
                    "command": "qdb"
                }

            # Handle list parameter - if True, use default 20
            list_limit = list if isinstance(list, int) else (20 if list else None)

            # Execute command
            if info:
                return self._show_collection_info(store)
            elif list is not None:
                return self._list_all_tickets(store, limit=list_limit)
            elif ticket_id:
                return self._search_by_ticket_id(store, ticket_id)
            elif search:
                return self._search_by_text(store, search, top_k=top_k)
            elif filter:
                return self._filter_by_metadata(store, service=service, severity=severity)
            else:
                return {
                    "error": "No command specified. Use --info, --list, --ticket-id, --search, or --filter",
                    "command": "qdb"
                }

        except Exception as e:
            if self.debug:
                raise
            return {
                "error": str(e),
                "command": "qdb"
            }

    def _show_collection_info(self, store) -> Dict[str, Any]:
        """Show collection statistics."""
        count = store.count_points()
        points = store.get_all_points()

        # Get unique ticket IDs
        ticket_ids = set()
        for point in points:
            payload = point.get('payload') if isinstance(point, dict) else point.payload
            ticket_id = payload.get("ticket_id", "N/A")
            if ticket_id and ticket_id != "N/A":
                ticket_ids.add(ticket_id)

        # Count by service
        service_counts = {}
        for point in points:
            payload = point.get('payload') if isinstance(point, dict) else point.payload
            service = payload.get('service', 'N/A')
            service_counts[service] = service_counts.get(service, 0) + 1

        return {
            "command": "info",
            "collection_name": store.collection_name,
            "total_points": count,
            "unique_tickets": len(ticket_ids),
            "host": store.host,
            "port": store.port,
            "dimensions": store.dimensions,
            "service_counts": service_counts
        }

    def _list_all_tickets(self, store, limit: int = 20) -> Dict[str, Any]:
        """List all unique ticket IDs in the database."""
        points = store.get_all_points()
        ticket_ids = set()

        for point in points:
            payload = point.get('payload') if isinstance(point, dict) else point.payload
            ticket_id = payload.get("ticket_id", "N/A")
            if ticket_id and ticket_id != "N/A":
                ticket_ids.add(ticket_id)

        sorted_tickets = sorted(list(ticket_ids))[:limit]

        return {
            "command": "list",
            "total_unique_tickets": len(ticket_ids),
            "showing": len(sorted_tickets),
            "tickets": sorted_tickets
        }

    def _search_by_ticket_id(self, store, ticket_id: str) -> Dict[str, Any]:
        """Search for all chunks belonging to a specific ticket ID."""
        # Simplified approach - get all points first, then try to get vector for each
        try:
            points = store.get_all_points()
            matching_chunks = []

            for point in points:
                payload = point.get('payload') if isinstance(point, dict) else point.payload
                if payload.get("ticket_id") == ticket_id:
                    point_id = str(point.get('id') if isinstance(point, dict) else point.id)

                    # Try to get the point with vector
                    try:
                        retrieved = store.client.retrieve(
                            collection_name=store.collection_name,
                            ids=[point_id],
                            with_payload=True,
                            with_vectors=True
                        )
                        if retrieved:
                            point_data = retrieved[0]
                            vector = point_data.vector if hasattr(point_data, 'vector') else None
                            payload_data = point_data.payload if hasattr(point_data, 'payload') else payload

                            # Check if vector is valid
                            vector_valid = vector is not None and len(vector) > 0
                            vector_norm = sum([x*x for x in vector]) ** 0.5 if vector_valid else 0

                            matching_chunks.append({
                                "id": point_id,
                                "chunk_index": payload_data.get('chunk_index', 'N/A'),
                                "service": payload_data.get('service', 'N/A'),
                                "severity": payload_data.get('severity', 'N/A'),
                                "text_preview": payload_data.get('chunk_text', '')[:200],
                                "full_text": payload_data.get('chunk_text', ''),
                                "vector_valid": vector_valid,
                                "vector_dim": len(vector) if vector_valid else 0,
                                "vector_norm": round(vector_norm, 4) if vector_valid else 0
                            })
                        else:
                            matching_chunks.append({
                                "id": point_id,
                                "chunk_index": payload.get('chunk_index', 'N/A'),
                                "service": payload.get('service', 'N/A'),
                                "severity": payload.get('severity', 'N/A'),
                                "text_preview": payload.get('chunk_text', '')[:200],
                                "full_text": payload.get('chunk_text', ''),
                                "vector_valid": "not_found",
                                "vector_dim": 0,
                                "vector_norm": 0
                            })
                    except Exception as e:
                        matching_chunks.append({
                            "id": point_id,
                            "chunk_index": payload.get('chunk_index', 'N/A'),
                            "service": payload.get('service', 'N/A'),
                            "severity": payload.get('severity', 'N/A'),
                            "text_preview": payload.get('chunk_text', '')[:200],
                            "full_text": payload.get('chunk_text', ''),
                            "vector_valid": f"error: {str(e)}",
                            "vector_dim": 0,
                            "vector_norm": 0
                        })

            return {
                "command": "ticket-id",
                "ticket_id": ticket_id,
                "found": len(matching_chunks) > 0,
                "chunk_count": len(matching_chunks),
                "chunks": matching_chunks
            }

        except Exception as e:
            # Fallback to scroll method without vectors
            points = store.get_all_points()
            matching_chunks = []

            for point in points:
                payload = point.get('payload') if isinstance(point, dict) else point.payload
                if payload.get("ticket_id") == ticket_id:
                    matching_chunks.append({
                        "id": str(point.get('id') if isinstance(point, dict) else point.id),
                        "chunk_index": payload.get('chunk_index', 'N/A'),
                        "service": payload.get('service', 'N/A'),
                        "severity": payload.get('severity', 'N/A'),
                        "text_preview": payload.get('chunk_text', '')[:200],
                        "full_text": payload.get('chunk_text', ''),
                        "vector_valid": "unknown",
                        "vector_dim": 0,
                        "vector_norm": 0
                    })

            return {
                "command": "ticket-id",
                "ticket_id": ticket_id,
                "found": len(matching_chunks) > 0,
                "chunk_count": len(matching_chunks),
                "chunks": matching_chunks,
                "error": str(e) if matching_chunks else None
            }

        if not matching_chunks:
            return {
                "command": "ticket-id",
                "ticket_id": ticket_id,
                "found": False,
                "chunks": []
            }

        return {
            "command": "ticket-id",
            "ticket_id": ticket_id,
            "found": True,
            "chunk_count": len(matching_chunks),
            "chunks": matching_chunks
        }

    def _search_by_text(self, store, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search by text similarity using embeddings."""
        try:
            from openai import OpenAI
            from supportdesk_rag.model_selector import get_embedding_model_name, get_client_kwargs

            client = OpenAI(**get_client_kwargs())
            model = get_embedding_model_name()
            response = client.embeddings.create(
                input=query,
                model=model
            )
            query_vector = response.data[0].embedding

        except Exception as e:
            return {
                "command": "search",
                "query": query,
                "error": f"Failed to generate embedding: {e}",
                "results": []
            }

        # Search
        results = store.search(query_vector, top_k=top_k)

        formatted_results = []
        for result in results:
            payload = result.get('payload') if isinstance(result, dict) else result.payload
            formatted_results.append({
                "score": result.get('score') if isinstance(result, dict) else result.score,
                "ticket_id": payload.get('ticket_id', 'N/A'),
                "service": payload.get('service', 'N/A'),
                "severity": payload.get('severity', 'N/A'),
                "text_preview": payload.get('chunk_text', '')[:200]
            })

        return {
            "command": "search",
            "query": query,
            "top_k": top_k,
            "result_count": len(formatted_results),
            "results": formatted_results
        }

    def _filter_by_metadata(self, store, service: str = None, severity: str = None) -> Dict[str, Any]:
        """Filter tickets by metadata fields."""
        points = store.get_all_points()
        filtered = []

        for point in points:
            payload = point.get('payload') if isinstance(point, dict) else point.payload
            match = True
            if service and payload.get('service') != service:
                match = False
            if severity and payload.get('severity') != severity:
                match = False
            if match:
                filtered.append(point)

        if not filtered:
            return {
                "command": "filter",
                "service": service,
                "severity": severity,
                "found": False,
                "chunks": []
            }

        # Group by ticket ID
        ticket_groups = {}
        for point in filtered:
            payload = point.get('payload') if isinstance(point, dict) else point.payload
            ticket_id = payload.get('ticket_id', 'N/A')
            if ticket_id not in ticket_groups:
                ticket_groups[ticket_id] = []
            ticket_groups[ticket_id].append(point)

        formatted_groups = []
        for ticket_id, chunks in ticket_groups.items():
            first_payload = chunks[0].get('payload') if isinstance(chunks[0], dict) else chunks[0].payload
            formatted_groups.append({
                "ticket_id": ticket_id,
                "chunk_count": len(chunks),
                "service": first_payload.get('service', 'N/A'),
                "severity": first_payload.get('severity', 'N/A')
            })

        return {
            "command": "filter",
            "service": service,
            "severity": severity,
            "found": True,
            "total_chunks": len(filtered),
            "ticket_count": len(formatted_groups),
            "tickets": formatted_groups
        }

    def format_output(self, result: Dict[str, Any], output_format: str = "markdown") -> str:
        """Format the output for display."""
        if output_format == "json":
            import json
            return json.dumps(result, indent=2, default=str)

        # Markdown format
        lines = []

        if "error" in result:
            lines.append("# ❌ Error")
            lines.append("")
            lines.append(f"```text\n{result['error']}\n```")
            return "\n".join(lines)

        command = result.get("command", "")

        if command == "info":
            lines.append("# 📊 Collection Information")
            lines.append("")
            lines.append(f"- **Collection Name:** {result.get('collection_name', 'N/A')}")
            lines.append(f"- **Total Points:** {result.get('total_points', 0)}")
            lines.append(f"- **Unique Tickets:** {result.get('unique_tickets', 0)}")
            lines.append(f"- **Host:** {result.get('host', 'N/A')}")
            lines.append(f"- **Port:** {result.get('port', 'N/A')}")
            lines.append(f"- **Dimensions:** {result.get('dimensions', 'N/A')}")
            lines.append("")
            lines.append("## Tickets by Service")
            lines.append("")
            service_counts = result.get('service_counts', {})
            for service, count in sorted(service_counts.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"- {service}: {count}")

        elif command == "list":
            lines.append("# 📋 Ticket List")
            lines.append("")
            lines.append(f"- **Total Unique Tickets:** {result.get('total_unique_tickets', 0)}")
            lines.append(f"- **Showing:** {result.get('showing', 0)}")
            lines.append("")
            tickets = result.get('tickets', [])
            for ticket_id in tickets:
                lines.append(f"- {ticket_id}")

        elif command == "ticket-id":
            ticket_id = result.get('ticket_id', 'N/A')
            lines.append(f"# 🔍 Ticket: {ticket_id}")
            lines.append("")
            if not result.get('found', False):
                lines.append("❌ No chunks found for this ticket")
            else:
                lines.append(f"✅ Found {result.get('chunk_count', 0)} chunks")
                lines.append("")
                chunks = result.get('chunks', [])
                for i, chunk in enumerate(chunks, 1):
                    lines.append(f"--- Chunk {i} ---")
                    lines.append(f"- **ID:** {chunk.get('id', 'N/A')}")
                    lines.append(f"- **Chunk Index:** {chunk.get('chunk_index', 'N/A')}")
                    lines.append(f"- **Service:** {chunk.get('service', 'N/A')}")
                    lines.append(f"- **Severity:** {chunk.get('severity', 'N/A')}")
                    # Only show vector info for ticket-id searches (when vector_valid is present)
                    if 'vector_valid' in chunk and chunk.get('vector_valid') != 'unknown':
                        lines.append(f"- **Vector Valid:** {chunk.get('vector_valid', 'unknown')}")
                        lines.append(f"- **Vector Dim:** {chunk.get('vector_dim', 0)}")
                        lines.append(f"- **Vector Norm:** {chunk.get('vector_norm', 0)}")
                    lines.append(f"- **Text Preview:** {chunk.get('text_preview', '')}...")
                    if 'full_text' in chunk:
                        lines.append(f"- **Full Text:**")
                        lines.append(f"```")
                        lines.append(chunk.get('full_text', ''))
                        lines.append(f"```")
                    lines.append("")

        elif command == "search":
            query = result.get('query', 'N/A')
            lines.append(f"# 🔍 Search Results: '{query}'")
            lines.append("")
            lines.append(f"- **Top-K:** {result.get('top_k', 0)}")
            lines.append(f"- **Results Found:** {result.get('result_count', 0)}")
            lines.append("")
            results = result.get('results', [])
            for i, res in enumerate(results, 1):
                lines.append(f"--- Result {i} ---")
                lines.append(f"- **Score:** {res.get('score', 0):.4f}")
                lines.append(f"- **Ticket ID:** {res.get('ticket_id', 'N/A')}")
                lines.append(f"- **Service:** {res.get('service', 'N/A')}")
                lines.append(f"- **Severity:** {res.get('severity', 'N/A')}")
                lines.append(f"- **Text Preview:** {res.get('text_preview', '')}...")
                lines.append("")

        elif command == "filter":
            lines.append("# 🔍 Filter Results")
            lines.append("")
            filters = []
            if result.get('service'):
                filters.append(f"service={result.get('service')}")
            if result.get('severity'):
                filters.append(f"severity={result.get('severity')}")
            lines.append(f"- **Filters:** {', '.join(filters) if filters else 'None'}")
            lines.append("")
            if not result.get('found', False):
                lines.append("❌ No matching tickets found")
            else:
                lines.append(f"- **Total Chunks:** {result.get('total_chunks', 0)}")
                lines.append(f"- **Ticket Count:** {result.get('ticket_count', 0)}")
                lines.append("")
                tickets = result.get('tickets', [])
                for ticket in tickets:
                    lines.append(f"### Ticket: {ticket.get('ticket_id', 'N/A')}")
                    lines.append(f"- **Chunks:** {ticket.get('chunk_count', 0)}")
                    lines.append(f"- **Service:** {ticket.get('service', 'N/A')}")
                    lines.append(f"- **Severity:** {ticket.get('severity', 'N/A')}")
                    lines.append("")

        return "\n".join(lines)


def main():
    """Main entry point for the qdb command."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Query Qdrant database directly for debugging"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show collection information"
    )
    parser.add_argument(
        "--list", "-l",
        nargs='?',
        const=20,
        type=int,
        help="List ticket IDs (default: 20, or specify number)"
    )
    parser.add_argument(
        "--ticket-id", "-t",
        type=str,
        help="Search for specific ticket ID"
    )
    parser.add_argument(
        "--search", "-s",
        type=str,
        help="Search by text similarity"
    )
    parser.add_argument(
        "--filter", "-f",
        action="store_true",
        help="Filter by metadata"
    )
    parser.add_argument(
        "--service",
        type=str,
        help="Filter by service (use with --filter)"
    )
    parser.add_argument(
        "--severity",
        type=str,
        help="Filter by severity (use with --filter)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results for search"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit for list command"
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
        help="Enable debug mode"
    )

    args = parser.parse_args()

    command = RagQdbCommand(
        verbose=args.verbose,
        debug=args.debug
    )

    # Handle list parameter - if True (default behavior), use limit
    list_value = args.list if args.list is not True else args.limit

    result = command.run(
        info=args.info,
        list=list_value,
        ticket_id=args.ticket_id,
        search=args.search,
        filter=args.filter,
        service=args.service,
        severity=args.severity,
        top_k=args.top_k,
        limit=args.limit
    )

    output_format = "json" if args.json_output else "markdown"
    output = command.format_output(result, output_format)
    print(output)

    if "error" in result:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
