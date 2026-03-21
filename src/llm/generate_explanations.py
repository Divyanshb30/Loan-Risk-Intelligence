import json
import time
from pathlib import Path
from openai import OpenAI
from src.utils.config import load_config, get_project_root
from dotenv import load_dotenv
import os

load_dotenv()  # loads .env into environment

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


config     = load_config()
output_dir = Path(get_project_root()) / config["paths"]["outputs"]
client     = OpenAI()  # reads OPENAI_API_KEY from environment

SYSTEM_PROMPT = """You are a senior credit risk analyst at a fintech company.
You are given a machine learning model's output for a loan application:
- A default probability score
- The top SHAP feature attributions explaining the score
- The macroeconomic context at time of loan issuance

Write a concise, professional 2-3 sentence risk explanation that:
1. States the risk level and probability clearly
2. Identifies the primary driver(s) using the SHAP values as evidence
3. Contextualises the macro environment if it is a top driver
4. Uses specific numbers from the input — do not be vague

Style: analyst memo, not a chatbot. No bullet points. No headers.
Length: 2-3 sentences maximum. Be precise and direct."""

def build_user_prompt(record: dict) -> str:
    drivers = record["shap_drivers"][:3]
    macro   = record["macro_context"]
    
    driver_lines = "\n".join([
        f"  - {d['feature']}: SHAP={d['shap_value']:+.4f} "
        f"({'increases' if d['direction'] == 'increases_risk' else 'reduces'} risk), "
        f"value={d['raw_value']}"
        for d in drivers
    ])
    
    macro_lines = "\n".join([
        f"  - {k}: {v:+.4f}" for k, v in macro.items()
    ])

    return f"""Loan Risk Assessment:
- Default Probability: {record['default_prob']:.1%}
- Risk Tier: {record['risk_tier']}
- Issue Year: {record['issue_year']}
- Interest Rate: {record['int_rate']:.1%}
- Loan Grade: {record['grade']}

Top SHAP Drivers:
{driver_lines}

Macro Context:
{macro_lines}

Write the risk explanation:"""

def generate_explanations(
    input_path:  Path,
    output_path: Path,
    model:       str  = "gpt-4o",   # cheap — $0.15/1M tokens
    batch_size:  int  = 50,
    resume_from: int  = 0
):
    with open(input_path) as f:
        records = json.load(f)

    # Resume from checkpoint if interrupted
    results = []
    checkpoint = output_path.with_suffix(".checkpoint.json")
    if checkpoint.exists() and resume_from == 0:
        with open(checkpoint) as f:
            results = json.load(f)
        resume_from = len(results)
        print(f"Resuming from checkpoint at {resume_from}/{len(records)}")

    records_to_process = records[resume_from:]

    for i, record in enumerate(records_to_process):
        global_i = resume_from + i

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(record)}
                ],
                temperature=0.7,
                max_tokens=150
            )
            explanation = response.choices[0].message.content.strip()

        except Exception as e:
            print(f"  ERROR on record {global_i}: {e}")
            explanation = ""

        results.append({
            "loan_id":       record["loan_id"],
            "default_prob":  record["default_prob"],
            "risk_tier":     record["risk_tier"],
            "actual_default":record["actual_default"],
            "shap_drivers":  record["shap_drivers"],
            "macro_context": record["macro_context"],
            "issue_year":    record["issue_year"],
            "explanation":   explanation
        })

        if (i + 1) % 10 == 0:
            print(f"Progress: {global_i + 1}/{len(records)} | "
                  f"Last: {explanation[:80]}...")
            # Save checkpoint every 10 records
            with open(checkpoint, "w") as f:
                json.dump(results, f, indent=2)

        # Rate limit buffer — gpt-4o-mini allows 500 RPM
        time.sleep(0.15)

    # Save final output
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Clean up checkpoint
    if checkpoint.exists():
        checkpoint.unlink()

    print(f"\nDone. {len(results)} explanations saved to {output_path}")

    # Quick quality check
    empty = sum(1 for r in results if not r["explanation"])
    print(f"Empty explanations: {empty}/{len(results)}")
    if empty > 0:
        print("Re-run with resume_from=0 to retry failed records")

    return results

if __name__ == "__main__":
    generate_explanations(
        input_path  = output_dir / "shap_dataset_raw.json",
        output_path = output_dir / "shap_dataset_explained.json"
    )
