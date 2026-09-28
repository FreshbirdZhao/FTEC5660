#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    ### YOUR CODE HERE
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_deepseek import ChatDeepSeek

    extraction_instructions = """You are a meticulous forensic accountant reading one Hong Kong supermarket receipt.
Read the receipt image twice before answering. Return only one valid JSON object with exactly these four keys:
{{"amount_paid": "0.00", "subtotal": "0.00", "discount_total": "0.00", "without_discount": "0.00"}}

Rules:
- amount_paid is the final amount actually paid after the ROUNDING line. Use the final payment/settlement amount, not cash tendered, change, balance, savings, or the pre-rounding subtotal.
- subtotal is the receipt's SUBTOTAL after all discounts but before ROUNDING.
- discount_total is the sum of the absolute values of every discount, promotion, coupon, member, app, packaging-damage, and percentage-off line. Exclude ROUNDING, payment, tender, and change lines.
- without_discount must equal subtotal + discount_total. Independently check it against the sum of the original positive item prices.
- Read decimal points and minus signs carefully. Ignore dates, times, quantities, product codes, loyalty points, and percentages as monetary amounts.
- Use HKD, exactly two decimal places, no currency symbols, no thousands separators, no explanations, and no Markdown fences.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", extraction_instructions),
            MessagesPlaceholder(variable_name="receipt"),
        ]
    )
    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
        max_tokens=4096,
        timeout=120,
        max_retries=3,
    )
    return prompt | model


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    from langchain_core.messages import HumanMessage

    cent = Decimal("0.01")

    def receipt_message(path: Path, correction: str = "") -> HumanMessage:
        instruction = (
            f"Extract the four required totals from receipt {path.name}. "
            "Return the JSON object only."
        )
        if correction:
            instruction += f" The previous extraction was unreliable: {correction} Re-read the image and correct it."
        return HumanMessage(
            content=[
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": image_data_url(path)}},
            ]
        )

    def parse_extraction(value: Any) -> dict[str, Decimal]:
        text = response_text(value)
        object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if object_match is None:
            raise ValueError("response did not contain a JSON object")
        data = json.loads(object_match.group(0))

        extracted: dict[str, Decimal] = {}
        for key in ("amount_paid", "subtotal", "discount_total", "without_discount"):
            raw_value = data.get(key)
            money_match = re.fullmatch(
                r"\s*(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)\s*",
                str(raw_value),
                flags=re.IGNORECASE,
            )
            if money_match is None:
                raise ValueError(f"invalid {key}: {raw_value!r}")
            extracted[key] = Decimal(money_match.group(1).replace(",", "")).quantize(cent)

        if extracted["amount_paid"] < 0 or extracted["subtotal"] < 0:
            raise ValueError("paid amount and subtotal must be non-negative")
        if extracted["discount_total"] < 0:
            extracted["discount_total"] = -extracted["discount_total"]
        expected_without_discount = extracted["subtotal"] + extracted["discount_total"]
        if abs(extracted["without_discount"] - expected_without_discount) > cent:
            raise ValueError("without_discount does not equal subtotal plus discounts")
        if abs(extracted["amount_paid"] - extracted["subtotal"]) > Decimal("0.10"):
            raise ValueError("final payment is inconsistent with subtotal and rounding")
        extracted["without_discount"] = expected_without_discount.quantize(cent)
        return extracted

    inputs = [{"receipt": [receipt_message(path)]} for path in images]
    raw_results = chain.batch(inputs, config={"max_concurrency": 3}, return_exceptions=True)

    extractions: list[dict[str, Decimal]] = []
    for path, raw_result in zip(images, raw_results):
        result = raw_result
        last_error = "model request failed" if isinstance(result, Exception) else ""
        for attempt in range(3):
            if not isinstance(result, Exception):
                try:
                    extractions.append(parse_extraction(result))
                    break
                except (ValueError, InvalidOperation, json.JSONDecodeError) as error:
                    last_error = str(error)
            if attempt == 2:
                raise RuntimeError(f"Could not extract reliable totals from {path.name}: {last_error}")
            result = chain.invoke({"receipt": [receipt_message(path, last_error)]})

    total_paid = sum((item["amount_paid"] for item in extractions), Decimal("0.00"))
    total_without_discount = sum(
        (item["without_discount"] for item in extractions), Decimal("0.00")
    )
    return {
        QUERY_1: f"HK${total_paid.quantize(cent):.2f}",
        QUERY_2: f"HK${total_without_discount.quantize(cent):.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
