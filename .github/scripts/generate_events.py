#!/usr/bin/env python3
"""
Fetches UK politics news from RSS feeds, uses Gemini to rewrite them
in the "Since Sean Left" tracker style, and injects them into index.html.
"""

import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import feedparser
import requests
from google import genai

RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/politics/rss.xml",
    "https://www.theguardian.com/politics/rss",
    "https://feeds.skynews.com/feeds/rss/politics.xml",
]

HOURS_LOOKBACK = 8  # wider window than 4h to catch things between runs

INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "index.html")

STYLE_PROMPT = """\
You are a writer for "Since Sean Left" — a satirical UK politics tracker that chronicles \
the chaos of British politics since 7 February 2026 (when Keir Starmer's government started \
to implode). The tone is: dark comedy meets journalistic precision. Facts are presented \
matter-of-factly with strategic additions — dramatic metaphors, casual asides, emphasis \
through repetition. The humor comes from the situation, not from being jokey.

STYLE EXAMPLES (copy this exact tone):

1. Title: "Morgan McSweeney Resigns as Chief of Staff"
   Desc: "Starmer's chief of staff quit, taking responsibility for advising the PM to appoint Mandelson as ambassador despite the Epstein connections. The architect leaves the building."
   Tags: ["resignation", "crisis"]

2. Title: "91 Flood Warnings Across England"
   Desc: "The Environment Agency issued 91 flood warnings and 263 flood alerts. The Met Office confirmed rain had fallen every single day of 2026 in south-west England. Every. Single. Day."
   Tags: ["failure"]

3. Title: "British Airways: 16 Cancellations, 330+ Delays"
   Desc: "BA's worst operational day of 2026 at Heathrow T5. Over 330 flight delays and 16 cancellations, stranding up to 5,000 passengers. Not directly the government's fault but it's the vibe."
   Tags: ["failure"]

4. Title: "Labour Polling: 19% — Third Place"
   Desc: "YouGov has Labour at 19%, behind Reform (26%) and the Conservatives (18% but closing). An MRP projection has Reform winning 381 seats at a general election."
   Tags: ["polls"]

5. Title: "Tim Allan Resigns as Communications Chief"
   Desc: "Director of communications quit the day after McSweeney. Two top aides gone in 24 hours. The bunker empties."
   Tags: ["resignation", "crisis"]

AVAILABLE TAGS (use 1-3 per event):
scandal, uturn, resignation, broken-promise, failure, polls, economic, security, hypocrisy, crisis, press, rebellion

RULES:
- Only write about genuinely notable UK political events
- Titles should be concise, factual headlines (under 80 chars)
- Descriptions should be 1-3 sentences, fact-based with subtle editorial flair
- Use the dramatic closer technique sparingly (e.g. "The bunker empties.")
- Keep casual asides rare and sharp
- Do NOT editorialize heavily — let the absurdity speak for itself
- Do NOT make things up — stick to the facts from the news items provided
- Do NOT cover stories already in the tracker (see existing titles below)
- If none of the news items are notable enough, return NONE

TODAY'S DATE: {today}

EXISTING EVENT TITLES (do not duplicate these topics):
{existing_titles}

NEWS ITEMS TO CONSIDER:
{news_items}

Return a JSON array of 0-3 event objects, or the string NONE if nothing is worth adding.
Format:
[
  {{
    "date": "{today}",
    "title": "...",
    "desc": "...",
    "tags": ["...", "..."]
  }}
]

Return ONLY the JSON array or NONE, no other text.
"""


def fetch_rss_items():
    """Fetch recent items from all RSS feeds."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=HOURS_LOOKBACK)
    items = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SinceSeanLeft/1.0)"}

    for url in RSS_FEEDS:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
            for entry in feed.entries:
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    published = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)

                # Include items with no date (can't filter them) or recent items
                if published and published < cutoff:
                    continue

                title = entry.get("title", "").strip()
                summary = entry.get("summary", "").strip()
                # Strip HTML tags from summary
                summary = re.sub(r"<[^>]+>", "", summary)

                if title:
                    items.append({"title": title, "summary": summary})
        except Exception as e:
            print(f"Warning: Failed to fetch {url}: {e}", file=sys.stderr)

    # Deduplicate by title
    seen = set()
    unique = []
    for item in items:
        if item["title"] not in seen:
            seen.add(item["title"])
            unique.append(item)

    return unique


def extract_existing_titles(html):
    """Extract existing event titles from index.html."""
    titles = re.findall(r"title:\s*['\"](.+?)['\"]", html)
    return titles


def generate_events(news_items, existing_titles, today):
    """Use Gemini to generate new events from news items."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    news_text = "\n".join(
        f"- {item['title']}: {item['summary']}" for item in news_items
    )
    existing_text = "\n".join(f"- {t}" for t in existing_titles)

    prompt = STYLE_PROMPT.format(
        today=today,
        existing_titles=existing_text or "(none yet)",
        news_items=news_text or "(no recent items found)",
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    text = response.text.strip()

    if text.upper() == "NONE":
        return []

    # Clean markdown code fences if present
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        events = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse Gemini response: {e}", file=sys.stderr)
        print(f"Response was: {text}", file=sys.stderr)
        return []

    if not isinstance(events, list):
        return []

    # Validate each event
    valid_tags = {
        "scandal", "uturn", "resignation", "broken-promise", "failure",
        "polls", "economic", "security", "hypocrisy", "crisis", "press", "rebellion",
    }
    validated = []
    for event in events:
        if not all(k in event for k in ("date", "title", "desc", "tags")):
            continue
        event["tags"] = [t for t in event["tags"] if t in valid_tags]
        if not event["tags"]:
            event["tags"] = ["crisis"]
        # Escape single quotes for JS
        event["title"] = event["title"].replace("'", "\\'")
        event["desc"] = event["desc"].replace("'", "\\'")
        validated.append(event)

    return validated[:3]  # Max 3


def inject_events(html, new_events):
    """Inject new events into the events array in index.html."""
    event_strings = []
    for event in new_events:
        tags_str = ", ".join(f"'{t}'" for t in event["tags"])
        event_str = (
            f"            {{\n"
            f"                date: '{event['date']}',\n"
            f"                title: '{event['title']}',\n"
            f"                desc: '{event['desc']}',\n"
            f"                tags: [{tags_str}]\n"
            f"            }}"
        )
        event_strings.append(event_str)

    insert_block = ",\n".join(event_strings)

    # Insert right after "const events = ["
    html = html.replace(
        "const events = [",
        f"const events = [\n{insert_block},",
    )

    return html


def main():
    # Read index.html
    index_path = os.path.normpath(INDEX_PATH)
    with open(index_path, "r") as f:
        html = f.read()

    existing_titles = extract_existing_titles(html)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print("Fetching RSS feeds...")
    news_items = fetch_rss_items()
    print(f"Found {len(news_items)} recent news items")

    if not news_items:
        print("No recent news items found. Skipping.")
        return

    print(f"Sending {len(news_items)} items to Gemini...")
    new_events = generate_events(news_items, existing_titles, today)

    if not new_events:
        print("Gemini returned no new events. Skipping.")
        return

    print(f"Generated {len(new_events)} new events:")
    for event in new_events:
        print(f"  - {event['title']}")

    updated_html = inject_events(html, new_events)

    with open(index_path, "w") as f:
        f.write(updated_html)

    print("index.html updated successfully.")


if __name__ == "__main__":
    main()
