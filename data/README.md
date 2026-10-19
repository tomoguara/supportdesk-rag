# Support Ticket Data Generator

This directory contains synthetic support ticket data generated for RAG system testing.

## Data Files

| File | Description | Size |
|------|-------------|------|
| `synthetic_tickets.json` | 100 tickets in JSON format | ~92KB |

## Data Generator

The generator script creates realistic support ticket data with:

### Services (10 categories)
- Network Infrastructure
- User Access & Authentication
- Email & Collaboration
- Cloud Services (AWS/Azure/GCP)
- Database Services
- Application Support
- Security Operations
- Monitoring & Alerts
- Backup & Recovery
- Hardware Support

### Severity Distribution (Realistic)
- Critical: 10%
- High: 20%
- Medium: 40%
- Low: 30%

### Ticket Types
- Incident, Service Request, Bug Report, Feature Request, Security Issue, Performance Issue, Account Issue, Configuration Change

### Generated Fields
- `ticket_id` - Unique ID (e.g., INC176399, SRV000567)
- `title` - Concise issue description with severity prefix
- `description` - Detailed problem description with technical terms
- `ticket_type`, `service`, `severity`, `priority`, `status`
- `created_at`, `updated_at` - Timestamps
- `user` - User information (name, email, team, location)
- `tags` - Service tags
- `attachments` - Number of attachments
- `sla_minutes` - SLA commitment
- `assigned_to` - Assigned team/person
- `resolution_notes` - Resolution information

## Usage

```bash
# Generate CSV and JSON with 100 tickets from last 30 days
python src/supportdesk_rag/data_generator.py --csv --json -n 100 -d 30

# Generate only CSV
python src/supportdesk_rag/data_generator.py --csv -n 200 -d 90

# Generate only JSON
python src/supportdesk_rag/data_generator.py --json -n 50 -d 7
```

## Data Generation Logic

1. **Weighted severity selection**: Uses configured weights to create realistic distribution
2. **Service-specific descriptions**: Generates technical descriptions based on service type
3. **Realistic user data**: Creates varied user profiles with realistic names and teams
4. **Time-based generation**: Creates tickets from within the specified days-back window

## License

MIT - See main project LICENSE.
