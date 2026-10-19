"""
Support Ticket Data Generator

Generates realistic support ticket data for RAG system testing.
Creates both CSV and JSON formats with varied, realistic data.
"""

import csv
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import os


# Constants for realistic data generation
SERVICES = [
    "Network Infrastructure",
    "User Access & Authentication",
    "Email & Collaboration",
    "Cloud Services (AWS/Azure/GCP)",
    "Database Services",
    "Application Support",
    "Security Operations",
    "Monitoring & Alerts",
    "Backup & Recovery",
    "Hardware Support"
]

SEVERITIES = {
    "Critical": {
        "weight": 0.10,  # 10% of tickets
        "title_prefixes": ["DOWN", "BREACH", "CRITICAL", "P0", "EMERGENCY"],
        "impact": "Complete service outage or security breach"
    },
    "High": {
        "weight": 0.20,  # 20% of tickets
        "title_prefixes": ["SLOW", "ERROR", "FAILURE", "P1", "URGENT"],
        "impact": "Major functionality degraded"
    },
    "Medium": {
        "weight": 0.40,  # 40% of tickets
        "title_prefixes": ["ISSUE", "PROBLEM", "REQUEST", "P2", "NORMAL"],
        "impact": "Partial functionality affected"
    },
    "Low": {
        "weight": 0.30,  # 30% of tickets
        "title_prefixes": ["QUESTION", "INQUIRY", "HELP", "P3", "INFO"],
        "impact": "Minor issue or question"
    }
}

TICKET_TYPES = [
    "Incident",
    "Service Request",
    "Bug Report",
    "Feature Request",
    "Security Issue",
    "Performance Issue",
    "Account Issue",
    "Configuration Change"
]

# Common technical terms for realistic descriptions
TECH_TERMS = {
    "network": [
        "connection timeout", "DNS resolution failed", "packet loss",
        "firewall rule", "VPN tunnel", "network latency",
        "bandwidth congestion", "router configuration", "switch port"
    ],
    "authentication": [
        "SSO failure", "MFA authentication", "token expired",
        "LDAP sync", "Active Directory", "OAuth2 flow",
        "password reset", "session timeout", "auth header"
    ],
    "email": [
        "SMTP relay", "exchange online", "outlook web",
        "email delivery", "spam filter", "attachment size",
        "mail flow", "transport rule", "mailbox quota"
    ],
    "cloud": [
        "EC2 instance", "S3 bucket", "Lambda function",
        "Azure VM", "Kubernetes pod", "CloudFormation",
        "IAM role", "VPC configuration", "NAT gateway"
    ],
    "database": [
        "query timeout", "connection pool", "deadlock",
        "replication lag", "index missing", "disk space",
        "backup failure", "restore process", "schema change"
    ],
    "application": [
        "API endpoint", "500 error", "404 not found",
        "middleware failure", "cache invalidation", "session state",
        "request timeout", "response latency", "service mesh"
    ],
    "security": [
        "malware detection", "intrusion attempt", "access violation",
        "privilege escalation", "data exfiltration", "audit log",
        "certificate expiry", "encryption failure", "compliance"
    ],
    "monitoring": [
        "alert threshold", "dashboard metric", "log aggregation",
        "synthetic check", "distributed tracing", "error rate",
        "health check", "uptime monitoring", "performance baseline"
    ],
    "backup": [
        "backup job", "restore point", "incremental backup",
        "storage lifecycle", "disaster recovery", "snapshot",
        "volume snapshot", "backup verification", "recovery time"
    ],
    "hardware": [
        "server rack", "power supply", "fan failure",
        "disk replacement", "memory upgrade", "network card",
        "cable replacement", "switch replacement", "hardware RAID"
    ]
}

# Sample questions and issues for realistic descriptions
SAMPLE_QUESTIONS = [
    "How do I reset my password?",
    "Why can't I access the dashboard?",
    "The system is running very slowly",
    "I need to add a new user account",
    "Is there a scheduled maintenance window?",
    "The API is returning errors",
    "I lost my MFA device, how do I recover?",
    "Can we increase the storage quota?",
    "The email integration stopped working",
    "I'm getting timeout errors on the portal"
]

SAMPLE_ISSUES = [
    "Application is throwing 500 errors intermittently",
    "Database connection pool exhausted",
    "Network latency increased by 300%",
    "Authentication tokens expiring too quickly",
    "Email notifications not being delivered",
    "API rate limiting is too aggressive",
    "Search functionality returning incomplete results",
    "File upload failing with 413 error",
    "User session not persisting across requests",
    "Calendar sync breaking after DST change"
]

SAMPLE_REQUESTS = [
    "Request to increase API rate limit",
    "Need access to analytics dashboard",
    "Ask for documentation on feature X",
    "Request to add new team member",
    "Need explanation of billing statement",
    "Ask about upcoming feature release",
    "Request training session for new tool",
    "Need help understanding error logs",
    "Ask about SLA guarantees",
    "Request custom report generation"
]

# User names for variety
FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Quinn", "Avery", "Parker"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]


def generate_ticket_id() -> str:
    """Generate a realistic ticket ID like INC001234 or SRV000567"""
    ticket_type = random.choice(["INC", "SRV", "BUG", "REQ", "SEC", "PERF"])
    number = random.randint(1000, 999999)
    return f"{ticket_type}{number:06d}"


def generate_timestamp(days_back: int = 30) -> str:
    """Generate a random timestamp within the last N days"""
    days = random.randint(0, days_back)
    hours = random.randint(0, 23)
    minutes = random.randint(0, 59)

    base_date = datetime.now() - timedelta(days=days)
    timestamp = base_date.replace(hour=hours, minute=minutes, second=0, microsecond=0)

    return timestamp.isoformat()


def generate_service_description(service: str, severity: str) -> str:
    """Generate a realistic description based on service and severity"""
    # Pick a category based on service
    service_keywords = {
        "Network Infrastructure": "network",
        "User Access & Authentication": "authentication",
        "Email & Collaboration": "email",
        "Cloud Services (AWS/Azure/GCP)": "cloud",
        "Database Services": "database",
        "Application Support": "application",
        "Security Operations": "security",
        "Monitoring & Alerts": "monitoring",
        "Backup & Recovery": "backup",
        "Hardware Support": "hardware"
    }

    keyword = service_keywords.get(service, "application")
    terms = TECH_TERMS[keyword]

    # Build description
    if severity in ["Critical", "High"]:
        issue_type = random.choice(SAMPLE_ISSUES)
        details = random.choice(terms)
        description = f"Issue: {issue_type}. Environment: {service}. Additional details: {details}. "
    else:
        question_type = random.choice(SAMPLE_QUESTIONS if severity == "Low" else SAMPLE_REQUESTS)
        details = random.choice(terms)
        description = f"Request: {question_type}. Context: {service}. Technical details: {details}. "

    # Add some variability
    description += random.choice([
        "This is affecting our production environment.",
        "We need this resolved as soon as possible.",
        "Please let us know if you need additional information.",
        "The issue started occurring today morning.",
        "This is a recurring problem that needs investigation."
    ])

    return description


def generate_user() -> Dict[str, str]:
    """Generate a random user"""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    return {
        "name": f"{first} {last}",
        "email": f"{first.lower()}.{last.lower()}@company.com",
        "team": random.choice(["Engineering", "Sales", "Marketing", "Support", "Finance", "HR", "Operations"]),
        "location": random.choice(["New York", "San Francisco", "Austin", "Seattle", "Boston", "Remote"])
    }


def generate_ticket(days_back: int = 30) -> Dict[str, Any]:
    """Generate a single realistic support ticket"""
    severity = random.choices(
        list(SEVERITIES.keys()),
        weights=[SEVERITIES[s]["weight"] for s in SEVERITIES.keys()]
    )[0]

    service = random.choice(SERVICES)
    ticket_type = random.choice(TICKET_TYPES)
    user = generate_user()

    # Generate title with appropriate prefix
    prefix = random.choice(SEVERITIES[severity]["title_prefixes"])
    title = f"[{prefix}] {service}: {random.choice(['Issue', 'Problem', 'Request', 'Question'])} - {random.choice(TECH_TERMS['application'])}"

    ticket = {
        "ticket_id": generate_ticket_id(),
        "title": title,
        "description": generate_service_description(service, severity),
        "ticket_type": ticket_type,
        "service": service,
        "severity": severity,
        "priority": random.choice(["P1", "P2", "P3", "P4"]),
        "status": random.choice(["New", "In Progress", "Pending", "Resolved", "Closed"]),
        "created_at": generate_timestamp(days_back),
        "updated_at": generate_timestamp(days_back),
        "user": user,
        "tags": random.sample(
            ["production", "staging", "urgent", "documentation", "integration", "api", "database", "security"],
            k=random.randint(1, 3)
        ),
        "attachments": random.randint(0, 3),  # Number of attachments
        "sla_minutes": random.choice([30, 60, 120, 240, 480, 1440]),  # SLA in minutes
        "assigned_to": random.choice(["John Doe", "Jane Smith", "Dev Team", "Support Team", "On-call Engineer", "Security Team"]),
        "resolution_notes": random.choice([
            None,
            "Fixed by updating the configuration.",
            "Resolved by restarting the service.",
            "User education provided.",
            "Ticket closed as resolved.",
            "Root cause identified, fix in progress."
        ])
    }

    return ticket


def generate_tickets_csv(num_tickets: int = 100, output_path: str = "data/synthetic_tickets.json", days_back: int = 30) -> str:
    """Generate tickets and save to CSV"""
    fieldnames = [
        "ticket_id", "title", "description", "ticket_type", "service",
        "severity", "priority", "status", "created_at", "updated_at",
        "user_name", "user_email", "user_team", "user_location",
        "tags", "attachments", "sla_minutes", "assigned_to", "resolution_notes"
    ]

    tickets = [generate_ticket(days_back) for _ in range(num_tickets)]

    # Flatten user data for CSV
    rows = []
    for ticket in tickets:
        user = ticket.pop("user")
        row = ticket.copy()
        row["user_name"] = user["name"]
        row["user_email"] = user["email"]
        row["user_team"] = user["team"]
        row["user_location"] = user["location"]
        rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def generate_tickets_json(num_tickets: int = 100, output_path: str = "data/synthetic_tickets.json", days_back: int = 30) -> str:
    """Generate tickets and save to JSON"""
    tickets = [generate_ticket(days_back) for _ in range(num_tickets)]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"tickets": tickets, "metadata": {
            "generated_at": datetime.now().isoformat(),
            "count": len(tickets),
            "days_back": days_back
        }}, f, indent=2, default=str)

    return output_path


def print_statistics(tickets: List[Dict[str, Any]]) -> None:
    """Print statistics about generated tickets"""
    services = {}
    severities = {}
    types = {}

    for ticket in tickets:
        service = ticket["service"]
        severity = ticket["severity"]
        ticket_type = ticket["ticket_type"]

        services[service] = services.get(service, 0) + 1
        severities[severity] = severities.get(severity, 0) + 1
        types[ticket_type] = types.get(ticket_type, 0) + 1

    print("\n" + "=" * 60)
    print("GENERATED TICKET STATISTICS")
    print("=" * 60)

    print("\nBy Service:")
    for service, count in sorted(services.items(), key=lambda x: x[1], reverse=True):
        print(f"  {count:3d} - {service}")

    print("\nBy Severity:")
    for severity, count in sorted(severities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {count:3d} - {severity} ({SEVERITIES[severity]['weight']*100:.0f}% weight)")

    print("\nBy Type:")
    for ticket_type, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {count:3d} - {ticket_type}")

    print("\n" + "=" * 60)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic support ticket data for RAG testing")
    parser.add_argument("--csv", action="store_true", help="Generate CSV output")
    parser.add_argument("--json", action="store_true", help="Generate JSON output")
    parser.add_argument("-n", "--num-tickets", type=int, default=100, help="Number of tickets to generate")
    parser.add_argument("-d", "--days-back", type=int, default=30, help="Generate tickets from last N days")

    args = parser.parse_args()

    if not args.csv and not args.json:
        args.csv = True
        args.json = True

    # Use the correct project data directory (parent of src)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_path = os.path.dirname(script_dir)  # Go up to project root
    data_path = os.path.join(base_path, "data")
    os.makedirs(data_path, exist_ok=True)

    tickets = None

    if args.csv:
        csv_path = os.path.join(data_path, "synthetic_tickets.json")
        generate_tickets_csv(args.num_tickets, csv_path, args.days_back)
        print(f"Generated {args.num_tickets} tickets to CSV: {csv_path}")

    if args.json:
        json_path = os.path.join(data_path, "synthetic_tickets.json")
        generate_tickets_json(args.num_tickets, json_path, args.days_back)
        print(f"Generated {args.num_tickets} tickets to JSON: {json_path}")
        tickets = [generate_ticket(args.days_back) for _ in range(args.num_tickets)]
        print_statistics(tickets)


if __name__ == "__main__":
    main()
