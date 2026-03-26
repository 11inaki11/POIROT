# Using the POIROT Agent

**Purpose:** Construct the error vector space for your multi-agent system

---

## Overview

The **POIROT Agent** is a specialized AI agent that analyzes your multi-agent system and identifies all potential error locations. This creates the **error vector space** - an N-dimensional space where each dimension represents a possible failure point.

### What It Does

Given a textual description of your system, the POIROT Agent:
1. Identifies all agents, components, and external actors
2. Determines which elements could be sources of errors
3. Constructs an N-dimensional error vector where each dimension represents a potential error location
4. Provides descriptions explaining why each location is a potential error source

### Example Output

For a trading system with 3 agents:

```json
{
  "system_name": "Stock Trading Bot",
  "error_regions": [
    {
      "id": "x1",
      "name": "Portfolio Manager",
      "type": "agent",
      "description": "Makes final trading decisions. Errors could stem from flawed logic."
    },
    {
      "id": "x2",
      "name": "Risk Assessor",
      "type": "agent",
      "description": "Evaluates risk. Errors from incorrect risk models or calculation bugs."
    },
    {
      "id": "x3",
      "name": "Market Analyst",
      "type": "agent",
      "description": "Analyzes trends. Errors from faulty models or misinterpretation."
    }
  ],
  "error_vector_example": [1, 0, 1]
}
```

This means:
- `[1, 0, 1]` = Portfolio Manager error + Market Analyst error (no Risk Assessor error)
- `[0, 1, 0]` = Risk Assessor error only
- `[1, 1, 1]` = All three agents have errors

---

## Installation

### Requirements

```bash
pip install langchain langchain-google-genai python-dotenv
```

### Environment Setup

Create a `.env` file with your Gemini API key:

```bash
GOOGLE_API_KEY=your_api_key_here
```

---

## Basic Usage

### Step 1: Describe Your System

Create a textual description of your multi-agent system:

```python
system_description = """
MY WAREHOUSE ROBOT SYSTEM

Agents:
- Inventory Manager: Tracks stock levels and reorders
- Route Planner: Optimizes robot navigation paths
- Picker Robot Controller: Controls picking robots
- Quality Inspector: Inspects items before packing

Hardware:
- Mobile picking robots (physical)
- Barcode scanners
- Conveyor belts

External:
- Warehouse Management System (WMS)
- Human warehouse workers
"""
```

### Step 2: Run POIROT Agent

```python
from poirot_agent import POIROTAgent

# Initialize agent
agent = POIROTAgent(model="gemini-2.5-pro")

# Analyze system
result = agent.analyze_system(system_description)

# Print results
print(f"System: {result['system_name']}")
print(f"Error Dimensions: {len(result['error_regions'])}")

for region in result['error_regions']:
    print(f"\n{region['id']}: {region['name']}")
    print(f"  Type: {region['type']}")
    print(f"  Description: {region['description']}")
```

### Step 3: Save Results

```python
# Save to JSON file
agent.save_output(result, "my_system_error_space.json")
```

---

## Advanced Usage

### Ignoring Known Non-Issues

If certain components should NOT be considered error sources:

```python
ignore_list = [
    "The WMS is a third-party system assumed to be reliable",
    "Human workers are out of scope for this analysis"
]

result = agent.analyze_system(system_description, ignore_list=ignore_list)
```

### Using Different LLM Models

```python
# Use a different model
agent = POIROTAgent(model="gemini-1.5-pro")

# Or with constructor-level ignore list
agent = POIROTAgent(
    model="gemini-2.5-pro",
    vectors_to_ignore=["External API is reliable"]
)
```

---

## Integration with Database

After running the POIROT Agent, populate your database's `error_vectors` table:

```python
import sqlite3

# Load POIROT Agent results
with open("my_system_error_space.json") as f:
    result = json.load(f)

# Connect to database
conn = sqlite3.connect("my_system_poirot.db")
cursor = conn.cursor()

# Insert error dimensions
for i, region in enumerate(result["error_regions"]):
    cursor.execute("""
        INSERT INTO error_vectors (
            error_id, system_name, dimension_index,
            dimension_name, dimension_description
        ) VALUES (?, ?, ?, ?, ?)
    """, (
        region["id"],
        result["system_name"],
        i,
        region["name"],
        region["description"]
    ))

conn.commit()
conn.close()
```

---

## Output Format

### Success Response

```json
{
  "system_name": "Stock Trading Bot",
  "error_regions": [
    {
      "id": "x1",
      "name": "Portfolio Manager",
      "type": "agent",
      "description": "..."
    }
  ],
  "error_vector_example": [1, 0, 0]
}
```

### Error Response

If parsing fails:

```json
{
  "error": "Could not parse LLM response as JSON",
  "raw": "... raw LLM output ..."
}
```

---

## Best Practices

### 1. Be Specific in System Description

**Good:**
```
AGENT: Portfolio Manager
- Makes BUY/SELL decisions
- Uses tools: get_price, place_order
- Communicates with Risk Assessor
```

**Too vague:**
```
AGENT: Portfolio Manager
- Manages portfolio
```

### 2. Include All Components

Make sure to describe:
- ✅ All agents
- ✅ External APIs/services
- ✅ Hardware components
- ✅ Human actors
- ✅ Critical software modules

### 3. Use Ignore List for Known Reliables

If a component is known to be reliable:
```python
ignore_list = [
    "The database is managed by AWS RDS (assumed reliable)",
    "Payment API is PCI-compliant (out of scope)"
]
```

### 4. Review and Validate Output

Always review the POIROT Agent's output:
- Are all critical agents included?
- Are there too many or too few error regions?
- Do the descriptions make sense?

You can manually edit the JSON if needed.

---

## Examples

### Example 1: Medical System

```python
medical_description = """
TELEMEDICINE PLATFORM

Agents:
- Triage Nurse: Initial patient assessment
- General Practitioner: Primary diagnosis
- Specialist: Detailed examination
- Pharmacist: Medication management

Components:
- Video call system
- Electronic Health Records (EHR)
- Patient (human)
"""

agent = POIROTAgent()
result = agent.analyze_system(medical_description)

# Expected output: ~7 error dimensions
# (4 agents + video system + EHR + patient)
```

### Example 2: Customer Service Bot

```python
service_description = """
CUSTOMER SERVICE SYSTEM

Agents:
- Intent Classifier: Identifies customer intent
- FAQ Bot: Answers common questions
- Escalation Manager: Routes to human agents
- Sentiment Analyzer: Monitors customer satisfaction

External:
- Knowledge Base API
- CRM System
- Human support agents
"""

ignore_list = [
    "CRM System is enterprise-grade (assume reliable)",
    "Human agents are well-trained (out of scope)"
]

agent = POIROTAgent()
result = agent.analyze_system(service_description, ignore_list)

# Expected output: ~5-6 error dimensions
# (4 agents + Knowledge Base, CRM and humans ignored)
```

---

## Troubleshooting

### Issue: "Could not parse LLM response as JSON"

**Solution:**
- Check your API key is valid
- Ensure your system description is clear
- Try a more specific description
- Check the `raw` field in the error response

### Issue: Too many error regions identified

**Solution:**
- Use the `ignore_list` parameter
- Simplify your system description
- Combine related components in description

### Issue: Missing error regions

**Solution:**
- Make your system description more detailed
- Explicitly mention all agents and components
- Check if components were accidentally in ignore list

---

## Next Steps

After constructing the error vector space:

1. ✅ Save results to JSON file
2. ✅ Populate `error_vectors` table in database
3. ✅ Collect execution data from your system
4. ⏳ Run POIROT Phase 1 & 2 analysis (coming soon)

---

## API Reference

### Class: `POIROTAgent`

```python
POIROTAgent(model: str = "gemini-2.5-pro", 
            vectors_to_ignore: Optional[List[str]] = None)
```

**Parameters:**
- `model`: LLM model name
- `vectors_to_ignore`: List of components to exclude

**Methods:**

#### `analyze_system(system_description: str, ignore_list: Optional[List[str]] = None) -> Dict[str, Any]`

Analyzes system and returns error vector space.

**Parameters:**
- `system_description`: Textual description of your system
- `ignore_list`: Optional ignore list (overrides constructor)

**Returns:** Dictionary with `system_name`, `error_regions`, `error_vector_example`

#### `save_output(parsed: Dict[str, Any], file_path: str) -> bool`

Saves analysis results to JSON file.

**Parameters:**
- `parsed`: Result dictionary from `analyze_system()`
- `file_path`: Path to save JSON file

**Returns:** `True` if successful, `False` otherwise

---

**Ready to analyze your system? Start with the example:**

```bash
cd examples/
python run_poirot_agent_example.py
```
