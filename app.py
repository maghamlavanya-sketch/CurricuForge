import os
import requests
import json
import re
from flask import Flask, render_template, request
from dotenv import load_dotenv
try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# model can be overridden via environment (use a provider-enabled model)
# default to a widely-supported model so users don't hit `model_not_supported` errors.
# you can change this via export MODEL_ID=<your-model> or in the .env file.
DEFAULT_MODEL = "gpt-3.5-turbo"  # change if you have a specific supported model
MODEL_ID = os.getenv("MODEL_ID", DEFAULT_MODEL)
# a secondary fallback model we'll try automatically if the primary is rejected
FALLBACK_MODEL = "gpt-3.5-turbo"
# Router chat endpoint for fallback HTTP requests
ROUTER_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

# Build headers each time so we can re-read the token if it changes (useful for tests)
def build_headers():
    hdrs = {"Content-Type": "application/json"}
    token = os.getenv("HUGGINGFACE_API_KEY")
    if token:
        hdrs["Authorization"] = f"Bearer {token}"
    return hdrs

# validate token early so we catch configuration issues quickly
if not HF_TOKEN:
    # the app will still start, but any model calls will return an explanatory message
    print("Warning: HUGGINGFACE_API_KEY not set. External requests will likely fail.\n"
          "Set the variable in your environment or .env file to avoid 401 errors.")

# Initialize huggingface_hub client if available
hf_client = None
if InferenceClient is not None:
    try:
        hf_client = InferenceClient(api_key=HF_TOKEN) if HF_TOKEN else InferenceClient()
    except Exception:
        hf_client = None


def try_parse_json(text):
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    s = text
    # If the text contains the key 'timetable', try to extract the JSON object around it
    key = 'timetable'
    key_idx = s.find(f'"{key}"')
    if key_idx == -1:
        key_idx = s.find(key)
    if key_idx != -1:
        # find the nearest opening brace before the key
        start = s.rfind('{', 0, key_idx)
        if start != -1:
            # find matching closing brace
            depth = 0
            for i in range(start, len(s)):
                if s[i] == '{':
                    depth += 1
                elif s[i] == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        candidate = s[start:end]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            break

    # fallback: find first { or [ and try substrings (brute-force)
    start_candidates = [i for i in (s.find('{'), s.find('[')) if i != -1]
    if not start_candidates:
        return None
    start = min(start_candidates)
    for end in range(len(s), start, -1):
        try:
            candidate = s[start:end]
            return json.loads(candidate)
        except Exception:
            continue
    return None


def parse_text_to_table(text):
    # Heuristic parser: find time ranges and split entries around them.
    if not isinstance(text, str):
        return None
    s = text.replace('\r', '')
    # Normalize separators
    s = s.replace('\t', ' ')

    time_re = re.compile(r"\d{1,2}:\d{2}\s*[-–—]\s*\d{1,2}:\d{2}")
    matches = list(time_re.finditer(s))
    entries = []
    if matches:
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i+1].start() if i+1 < len(matches) else len(s)
            block_start = s.rfind('\n\n', 0, start)
            if block_start == -1:
                block_start = 0
            chunk = s[block_start:end].strip()
            # Extract period (look for 'Day' or first line)
            period = ''
            # Look backward from time for 'Day' phrase
            pre = s[block_start:start]
            day_match = re.search(r"(Day\s*\d[^\n\r\-:]*)", pre)
            if day_match:
                period = day_match.group(1).strip()
            else:
                # take first non-empty line
                lines = chunk.splitlines()
                period = lines[0].strip() if lines else ''
            time_str = m.group(0).strip()
            # Activity: text after the time in the chunk
            after = s[m.end():end].strip()
            # Remove leading punctuation
            after = re.sub(r"^[\-–—:\s]+", '', after)
            # Collapse multiple spaces/newlines
            activity = ' '.join(after.split())
            if activity:
                entries.append({"period": period, "time": time_str, "activity": activity})
    else:
        # Fallback: split by double newlines into blocks
        blocks = [b.strip() for b in s.split('\n\n') if b.strip()]
        for b in blocks:
            lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
            if not lines:
                continue
            period = lines[0]
            time_str = ''
            activity = ' '.join(lines[1:]) if len(lines) > 1 else ''
            entries.append({"period": period, "time": time_str, "activity": activity})

    return entries if entries else None


def build_weekly_grid(source):
    """Convert various model outputs into a weekly grid dict with keys `days` and `rows`.
    `source` may be a dict (with rows/days), a list of entries, or a text string.
    """
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    # Normalize to list of entries with period/time/activity
    entries = None
    if isinstance(source, dict) and 'rows' in source and 'days' in source:
        return {"days": source['days'], "rows": source['rows']}
    if isinstance(source, list):
        entries = source
    elif isinstance(source, dict) and 'timetable' in source and isinstance(source['timetable'], list):
        entries = source['timetable']
    elif isinstance(source, str):
        # try to parse JSON then text
        parsed = try_parse_json(source)
        if isinstance(parsed, dict) and 'timetable' in parsed:
            entries = parsed['timetable']
        elif isinstance(parsed, dict) and 'rows' in parsed and 'days' in parsed:
            return {"days": parsed['days'], "rows": parsed['rows']}
        elif isinstance(parsed, list):
            entries = parsed
        else:
            entries = parse_text_to_table(source)

    if not entries:
        # fallback: create empty grid with default slots
        default_times = ["08:50-09:05","09:05-09:35","09:35-09:40","09:40-10:20","10:20-11:00","11:00-11:20","11:20-12:05","12:05-12:25","12:25-12:40","12:40-13:30","13:30-14:30","14:30-16:00"]
        rows = []
        for t in default_times:
            row = {"time": t}
            for d in days:
                row[d] = ""
            rows.append(row)
        return {"days": days, "rows": rows}

    # Determine time slots from entries if available
    times = []
    for e in entries:
        t = e.get('time') if isinstance(e, dict) else None
        if t:
            t = t.strip()
            if t and t not in times:
                times.append(t)

    # if not enough unique times, use default compact slots
    if len(times) < 5:
        times = ["09:00-10:30", "10:30-11:15", "11:30-12:30", "14:00-15:00", "15:15-16:00"]

    # initialize rows
    rows = []
    for t in times:
        row = {"time": t}
        for d in days:
            row[d] = ""
        rows.append(row)

    # helper to map Day N to weekday
    def day_from_period(period):
        if not period:
            return None
        m = re.search(r"Day\s*(\d+)", period, re.IGNORECASE)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(days):
                return days[idx]
        # try matching day names
        for d in days:
            if d.lower() in period.lower():
                return d
        return None

    # fill rows with entries
    # keep counters for each day to place items sequentially
    day_counters = {d: 0 for d in days}
    for e in entries:
        if not isinstance(e, dict):
            continue
        period = e.get('period', '')
        activity = e.get('activity', '')
        time_str = e.get('time', '')
        day = day_from_period(period)
        if not day:
            # assign by round-robin to next day with available slot
            # pick day by minimal count
            day = min(day_counters, key=lambda k: day_counters[k])

        # find row index
        row_idx = None
        if time_str:
            # try exact match
            for i, r in enumerate(rows):
                if time_str in r['time'] or r['time'] in time_str or time_str == r['time']:
                    row_idx = i
                    break
        if row_idx is None:
            # use next available slot for this day
            row_idx = day_counters[day] % len(rows)

        # append or set activity
        existing = rows[row_idx].get(day, '')
        if existing:
            rows[row_idx][day] = existing + "; " + activity
        else:
            rows[row_idx][day] = activity

        day_counters[day] += 1

    return {"days": days, "rows": rows}


def convert_to_triplets(source):
    """Return list of {topic, day, time} from various source shapes."""
    triplets = []
    # If it's a grid
    if isinstance(source, dict) and 'rows' in source and 'days' in source:
        days = source['days']
        for row in source['rows']:
            time = row.get('time', '')
            for d in days:
                val = row.get(d, '')
                if val and val.strip():
                    triplets.append({
                        'topic': val.strip(),
                        'day': d,
                        'time': time
                    })
        return triplets

    # If it's a list of entries
    entries = None
    if isinstance(source, list):
        entries = source
    elif isinstance(source, dict) and 'timetable' in source:
        entries = source['timetable']
    elif isinstance(source, str):
        parsed = try_parse_json(source)
        if isinstance(parsed, dict) and ('rows' in parsed and 'days' in parsed):
            return convert_to_triplets(parsed)
        if isinstance(parsed, dict) and 'timetable' in parsed:
            entries = parsed['timetable']
        elif isinstance(parsed, list):
            entries = parsed
        else:
            entries = parse_text_to_table(source)

    if entries:
        for e in entries:
            if not isinstance(e, dict):
                continue
            topic = e.get('activity') or e.get('activity', '') or e.get('topic') or ''
            time = e.get('time', '')
            period = e.get('period', '')
            # determine day from period
            day = None
            m = re.search(r"Day\s*(\d+)", period or '', re.IGNORECASE)
            if m:
                idx = int(m.group(1)) - 1
                days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                if 0 <= idx < len(days):
                    day = days[idx]
            else:
                # check for weekday name
                for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
                    if d.lower() in (period or '').lower():
                        day = d
                        break
            if not day:
                day = ''
            if topic and topic.strip():
                triplets.append({'topic': topic.strip(), 'day': day, 'time': time})

    return triplets

def generate_timetable(subject, topics_str, free_time, plan_type):
    """Generate a timetable structure using the user's inputs.

    Split the available time into fixed one‑hour slots starting at 09:00, and
    assign the study topic to each slot. Output shapes match the existing
    template: a list for daily/monthly or a grid for weekly.
    """
    subj = subject.strip() if isinstance(subject, str) else "Study"
    topics = []
    if isinstance(topics_str, str):
        topics = [t.strip() for t in topics_str.split(',') if t.strip()]
    if not topics:
        topics = [subj]
    ft = free_time.strip() if isinstance(free_time, str) else ""
    plan = plan_type.strip().lower() if isinstance(plan_type, str) else ""

    def topic_label(idx):
        return topics[idx % len(topics)]

    # parse number of hours from the free_time string
    import re
    m = re.search(r"(\d+)", ft)
    hours = int(m.group(1)) if m else 1

    # determine minutes available for study after reserving breaks
    n_topics = len(topics)
    # each inter-topic break is 10 minutes; there are n_topics-1 breaks
    break_minutes = max(0, (n_topics - 1) * 10)
    total_available = hours * 60
    total_study = max(0, total_available - break_minutes)

    base_minutes = total_study // n_topics if n_topics else 0
    extra = total_study - base_minutes * n_topics

    # helper to format a span
    def format_span(start_min, end_min):
        h1, m1 = divmod(start_min, 60)
        h2, m2 = divmod(end_min, 60)
        return f"{h1:02d}:{m1:02d}-{h2:02d}:{m2:02d}"

    raw_times = []  # list of tuples (time_str, is_break, topic_index)
    current = 9 * 60
    for i in range(n_topics):
        duration = base_minutes + (1 if i < extra else 0)
        start = current
        end = start + duration
        raw_times.append((format_span(start, end), False, i))
        current = end
        if i < n_topics - 1:
            bstart = current
            bend = bstart + 10
            raw_times.append((format_span(bstart, bend), True, None))
            current = bend

    if plan == "daily":
        # create row for each entry (including breaks)
        return [
            {"period": f"Slot {i+1}",
             "time": raw_times[i][0],
             "activity": "Break" if raw_times[i][1] else topic_label(i // 2)}
            for i in range(len(raw_times))
        ]

    if plan == "weekly":
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        rows = []
        for entry in raw_times:
            t, is_break, topic_idx = entry
            row = {"time": t}
            for d in days:
                if is_break:
                    row[d] = "Break"
                else:
                    # use topic_idx to label, cycle per day if needed
                    row[d] = topic_label(topic_idx)
            rows.append(row)
        return {"days": days, "rows": rows}

    if plan == "monthly":
        # show each topic with its allocated time (ignore breaks)
        result = []
        study_slots = [(t, idx) for t, br, idx in raw_times if not br]
        for w, (t, idx) in enumerate(study_slots[:4]):
            result.append({
                "period": f"Week {w+1}",
                "time": t,
                "activity": topic_label(idx)
            })
        # if there are fewer than 4 topics, repeat as needed
        while len(result) < 4:
            idx = result[len(result) - 1]['activity']
            result.append(result[-1])
        return result

    # fallback: list slots with cyclic topics
    return [
        {"period": f"{plan_type} {i+1}",
         "time": slots[i % len(slots)],
         "activity": topic_label(i)}
        for i in range(max(1, len(slots)))
    ]


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    table = None
    grid = None

    if request.method == "POST":
        subject = request.form["subject"]
        topics = request.form.get("topics", "")
        free_time = request.form["free_time"]
        plan_type = request.form["plan_type"]

        result = generate_timetable(subject, topics, free_time, plan_type)
        if isinstance(result, dict) and 'rows' in result and 'days' in result:
            grid = result
        elif isinstance(result, list):
            table = result
        else:
            parsed = None
            if isinstance(result, dict):
                parsed = result
            elif isinstance(result, str):
                try:
                    parsed = json.loads(result)
                except Exception:
                    s = result
                    start_candidates = [i for i in (s.find('{'), s.find('[')) if i != -1]
                    if start_candidates:
                        start = min(start_candidates)
                        for end in range(len(s), start, -1):
                            try:
                                parsed = json.loads(s[start:end])
                                break
                            except Exception:
                                continue

            if isinstance(parsed, dict) and "timetable" in parsed and isinstance(parsed["timetable"], list):
                table = parsed["timetable"]

        # If table wasn't found but we can convert text to table, do it
        if not table and not grid and isinstance(result, str):
            tbl = parse_text_to_table(result)
            if tbl:
                table = tbl
    return render_template("index.html", result=result, table=table, grid=grid)


if __name__ == "__main__":
    app.run(debug=True)
