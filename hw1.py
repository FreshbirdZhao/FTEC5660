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
Return only one valid JSON object with exactly these keys:
{{"item_total": "0.00", "discounts": [{{"label": "short label", "amount": "0.00"}}], "subtotal": "0.00", "rounding": "0.00", "amount_paid": "0.00"}}

Definitions:
- item_total: the sum of every ORIGINAL positive item price printed on the receipt, BEFORE any discount is applied. Include each product line at its printed price and never subtract anything here.
- discounts: EVERY line that reduces the bill (promotion, coupon, member, app, percentage-off, packaging-damage, "X% OFF", "SAVE", "DISC"). Give one entry per printed discount line with the amount as a POSITIVE number. Never include ROUNDING, payment, tender, change, loyalty points, or the SUBTOTAL line.
- subtotal: the receipt's printed SUBTOTAL, the total after all discounts and before ROUNDING.
- rounding: the signed ROUNDING amount exactly as printed (for example -0.01 or 0.05); use 0.00 when the receipt has no rounding line.
- amount_paid: the final amount actually paid after rounding (the payment/settlement amount such as OCTOPUS/EPS/CASH), not cash tendered, change, balance, or savings.

Rules:
- Read the discount lines one by one; do not confuse the SUBTOTAL with a discount.
- Read decimal points and minus signs carefully.
- Ignore dates, times, quantities, product codes, loyalty points and percentages as monetary amounts.
- Exactly two decimal places, no currency symbols, no thousands separators, no explanations, no Markdown fences.
- Before answering, verify that item_total equals subtotal plus the sum of all discount amounts.
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
        max_tokens=16384,
        timeout=180,
        max_retries=3,
    )
    # This is a reasoning model: it emits a long hidden chain of thought before the
    # visible answer, and max_tokens is shared by both. A tight cap can therefore
    # truncate the reply before any JSON appears, so keep generous headroom and ask
    # for DeepSeek's JSON mode so the visible answer is always a JSON object.
    return prompt | model.bind(response_format={"type": "json_object"})


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
    tolerance = Decimal("0.02")

    def money(value: Any, field: str) -> Decimal:
        """Parse one monetary amount from the model's JSON."""
        money_match = re.fullmatch(
            r"\s*(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)\s*",
            str(value),
            flags=re.IGNORECASE,
        )
        if money_match is None:
            raise ValueError(f"invalid {field}: {value!r}")
        return Decimal(money_match.group(1).replace(",", "")).quantize(cent)

    def receipt_message(path: Path, correction: str = "") -> HumanMessage:
        instruction = (
            f"Analyse receipt {path.name} and return the JSON object only. "
            "Write down every discount line separately before you total them."
        )
        if correction:
            instruction += (
                f"\nThe previous attempt was rejected: {correction[:300]}"
                "\nRe-read the image carefully, fix that problem, and return the JSON object."
            )
        return HumanMessage(
            content=[
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": image_data_url(path)}},
            ]
        )

    def response_body(value: Any) -> str:
        """Visible text, falling back to hidden reasoning when the content is empty."""
        text = response_text(value)
        if "{" not in text:
            extra = getattr(value, "additional_kwargs", None)
            if isinstance(extra, dict):
                reasoning = extra.get("reasoning_content")
                if isinstance(reasoning, str) and "{" in reasoning:
                    text = f"{text}\n{reasoning}"
        return text

    def json_object(value: Any) -> dict:
        text = re.sub(r"```(?:json)?", "", response_body(value))
        for candidate in reversed(re.findall(r"\{.*\}", text, flags=re.DOTALL)):
            try:
                parsed_candidate = json.loads(candidate)
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(parsed_candidate, dict) and "amount_paid" in parsed_candidate:
                return parsed_candidate
        raise ValueError("response did not contain a usable JSON object")

    def parse_extraction(value: Any, strict: bool = True) -> dict[str, Decimal]:
        """Read the five-field JSON and cross-check it with independent arithmetic.

        A missed or misread discount line stays self-consistent in the old
        ``subtotal + discount_total`` scheme, so the model must also report the
        original item total: ``item_total == subtotal + sum(discounts)`` is an
        independent check that exposes a missing line.
        """
        data = json_object(value)
        subtotal = money(data.get("subtotal"), "subtotal")
        amount_paid = money(data.get("amount_paid"), "amount_paid")
        rounding = money(data.get("rounding", "0.00"), "rounding")

        raw_discounts = data.get("discounts")
        discount_lines: list[Decimal] = []
        if isinstance(raw_discounts, list):
            for entry in raw_discounts:
                amount = entry.get("amount") if isinstance(entry, dict) else entry
                discount_lines.append(abs(money(amount, "discount line")))
            discount_total = sum(discount_lines, Decimal("0.00")).quantize(cent)
        elif raw_discounts is None:
            discount_total = Decimal("0.00")
        else:
            discount_total = abs(money(raw_discounts, "discounts")).quantize(cent)

        raw_item_total = data.get("item_total")
        item_total = None if raw_item_total is None else money(raw_item_total, "item_total")

        if strict:
            if subtotal <= 0 or amount_paid <= 0:
                raise ValueError("subtotal and amount_paid must be positive")
            if abs(rounding) > Decimal("0.10"):
                raise ValueError(f"rounding {rounding} is too large")
            if abs(amount_paid - (subtotal + rounding)) > cent:
                raise ValueError(
                    f"amount_paid {amount_paid} does not equal subtotal {subtotal} "
                    f"plus rounding {rounding}"
                )
            if item_total is not None:
                expected_items = (subtotal + discount_total).quantize(cent)
                if abs(item_total - expected_items) > tolerance:
                    raise ValueError(
                        f"item_total {item_total} does not equal subtotal {subtotal} plus "
                        f"discounts {discount_total} = {expected_items}; a discount line "
                        "may be missing or misread"
                    )

        return {
            "amount_paid": amount_paid,
            "subtotal": subtotal,
            "discount_total": discount_total,
            "without_discount": (subtotal + discount_total).quantize(cent),
        }

    def lenient(value: Any) -> dict[str, Decimal] | None:
        try:
            return parse_extraction(value, strict=False)
        except (ValueError, InvalidOperation, json.JSONDecodeError):
            return None

    def interpret(value: Any) -> tuple[dict[str, Decimal] | None, str, dict[str, Decimal] | None]:
        if isinstance(value, Exception):
            return None, f"the model request failed ({type(value).__name__})", None
        try:
            return parse_extraction(value), "", None
        except (ValueError, InvalidOperation, json.JSONDecodeError) as error:
            return None, str(error), lenient(value)

    def reading_key(item: dict[str, Decimal]) -> tuple[Decimal, Decimal, Decimal]:
        return (item["amount_paid"], item["subtotal"], item["discount_total"])

    def agreed_reading(readings: list[dict[str, Decimal]]) -> dict[str, Decimal] | None:
        """Return a reading that at least two independent passes agree on."""
        if len(readings) < 2:
            return None
        counts: dict[tuple[Decimal, Decimal, Decimal], int] = {}
        for item in readings:
            key = reading_key(item)
            counts[key] = counts.get(key, 0) + 1
        best_key, best_count = max(counts.items(), key=lambda pair: pair[1])
        if best_count < 2:
            return None
        for item in readings:
            if reading_key(item) == best_key:
                return item
        return None

    def describe(item: dict[str, Decimal]) -> str:
        return (
            f"paid {item['amount_paid']}, subtotal {item['subtotal']}, "
            f"discounts {item['discount_total']}"
        )

    def read_once(path: Path, correction: str = "") -> Any:
        try:
            return chain.invoke({"receipt": [receipt_message(path, correction)]})
        except Exception as error:  # noqa: BLE001 - the runner must never abort
            return error

    # Read every receipt twice: a single internally consistent reading can still
    # be wrong, because the model may drop a discount line from both item_total
    # and the discount list. Two independent passes expose that disagreement.
    rounds = [
        chain.batch(
            [{"receipt": [receipt_message(path)]} for path in images],
            config={"max_concurrency": 3},
            return_exceptions=True,
        )
        for _ in range(2)
    ]

    extractions: dict[str, dict[str, Decimal]] = {}
    for index, path in enumerate(images):
        readings: list[dict[str, Decimal]] = []
        best_effort: dict[str, Decimal] | None = None
        correction = ""
        for round_results in rounds:
            parsed, error, fallback = interpret(round_results[index])
            if parsed is None:
                correction = error or correction
                print(f"warning: {path.name} read rejected ({error[:200]})")
            else:
                readings.append(parsed)
            if best_effort is None:
                best_effort = fallback

        attempt = 0
        while agreed_reading(readings) is None and attempt < 3:
            if len(readings) >= 2:
                correction = (
                    "two independent reads disagreed ("
                    + " ; ".join(describe(item) for item in readings[-2:])
                    + "); re-read the receipt line by line and correct the wrong one"
                )
            result = read_once(path, correction)
            parsed, error, fallback = interpret(result)
            if parsed is None:
                correction = error or correction
                print(f"warning: {path.name} extra read {attempt + 1} rejected ({error[:200]})")
            else:
                correction = ""
                readings.append(parsed)
            if best_effort is None:
                best_effort = fallback
            attempt += 1

        chosen = agreed_reading(readings) or (readings[0] if readings else None) or best_effort
        if chosen is None:
            # Never abort the run: results.csv must exist, so a receipt that gave
            # no usable numbers at all is counted as 0.00 and reported.
            print(f"warning: {path.name} unreadable after retries; counting it as 0.00")
            chosen = {
                "amount_paid": Decimal("0.00"),
                "subtotal": Decimal("0.00"),
                "discount_total": Decimal("0.00"),
                "without_discount": Decimal("0.00"),
            }
        elif chosen not in readings:
            print(f"warning: {path.name} kept its last usable reading (no agreement)")
        extractions[path.name] = chosen

    total_paid = sum((item["amount_paid"] for item in extractions.values()), Decimal("0.00"))
    total_without_discount = sum(
        (item["without_discount"] for item in extractions.values()), Decimal("0.00")
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
