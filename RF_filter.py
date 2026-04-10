import math
import time
from collections import deque


## CONFIG

NOVELTY_THRESHOLD    = 0.6       # detections scoring above this get queued
PRIORITY_LABELS      = {"radar", "satellite_downlink", "emergency_beacon"}
BASELINE_WINDOW_S    = 300.0     # rolling window for baseline stats (seconds)
MIN_CONFIDENCE       = 0.4       # drop anything below this confidence
MAX_QUEUE_DEPTH      = 50        # oldest entry dropped if queue is full
POWER_Z_THRESHOLD    = 2.0       # how many std devs counts as a power anomaly
FREQ_NOVELTY_FRAC    = 0.15      # fraction of unseen bins needed to count as novel
BIN_HZ               = 100_000   # frequency bin width: 100 kHz


## BASELINE HELPERS

def get_bin(freq_hz):
    """Snap a frequency to the nearest 100 kHz bin."""
    return int(freq_hz // BIN_HZ)


def expire_old_entries(history, bands, now):
    """Remove baseline entries older than the rolling window."""
    cutoff = now - BASELINE_WINDOW_S
    while history and history[0][0] < cutoff:
        ts, fbin = history.popleft()
        if fbin in bands:
            bands[fbin]["count"] = max(0, bands[fbin]["count"] - 1)
            if bands[fbin]["count"] == 0:
                del bands[fbin]


def update_band(bands, fbin, power_db, ts):
    """Add a new observation to a frequency band's running stats."""
    if fbin not in bands:
        bands[fbin] = {"count": 0, "power_sum": 0.0, "power_sq_sum": 0.0, "last_seen": 0.0}
    b = bands[fbin]
    b["count"]        += 1
    b["power_sum"]    += power_db
    b["power_sq_sum"] += power_db ** 2
    b["last_seen"]     = ts


def band_mean(b):
    return b["power_sum"] / b["count"] if b["count"] else 0.0


def band_std(b):
    if b["count"] < 2:
        return 0.0
    var = (b["power_sq_sum"] / b["count"]) - band_mean(b) ** 2
    return math.sqrt(max(var, 0.0))


def fraction_unseen(bands, freq_hz, bandwidth):
    """What fraction of the band around freq_hz has never been seen?"""
    lo = get_bin(freq_hz - bandwidth / 2)
    hi = get_bin(freq_hz + bandwidth / 2)
    n_bins = max(hi - lo + 1, 1)
    unseen = sum(1 for b in range(lo, hi + 1) if b not in bands)
    return unseen / n_bins


def ingest_detection(detection, bands, history):
    """Add a detection into the baseline store."""
    now  = detection["timestamp"]
    expire_old_entries(history, bands, now)
    fbin = get_bin(detection["freq_hz"])
    update_band(bands, fbin, detection["power_db"], now)
    history.append((now, fbin))


## NOVELTY SCORING

def compute_novelty(detection, bands):
    freq_hz   = detection["freq_hz"]
    power_db  = detection["power_db"]
    bandwidth = detection["bandwidth"]
    now       = detection["timestamp"]

    # Component 1: how much of the frequency range is unseen?
    freq_novel = fraction_unseen(bands, freq_hz, bandwidth)

    # Component 2: how anomalous is the power level?
    b = bands.get(get_bin(freq_hz))
    if b is None or b["count"] < 2:
        power_novel = 1.0   # never seen before — maximally novel
    else:
        z = abs(power_db - band_mean(b)) / max(band_std(b), 0.5)
        power_novel = min(z / POWER_Z_THRESHOLD, 1.0)

    # Component 3: how long since this freq was last seen?
    if b is None:
        recency_score = 1.0
    else:
        gap_s = now - b["last_seen"]
        recency_score = 1.0 / (1.0 + math.exp(-(gap_s - 30) / 10))

    return (freq_novel + power_novel + recency_score) / 3.0


## FILTER / QUEUE

def make_filter_state():
    """Create a fresh filter state dict (replaces the TransmissionFilter class)."""
    return {
        "bands":   {},                          # freq_bin -> band stats dict
        "history": deque(),                     # sliding window of (ts, bin)
        "queue":   deque(maxlen=MAX_QUEUE_DEPTH),
        "stats":   {"seen": 0, "queued": 0, "dropped": 0, "priority": 0},
    }


def process_detection(detection, state):
    """
    Run one detection through the filter.
    Returns a result dict with keys: detection, novelty_score, reason.
    """
    state["stats"]["seen"] += 1

    # Gate on minimum confidence
    if detection.get("confidence", 0) < MIN_CONFIDENCE:
        state["stats"]["dropped"] += 1
        return {"detection": detection, "novelty_score": 0.0, "reason": "dropped:low_confidence"}

    novelty = compute_novelty(detection, state["bands"])

    # Always update baseline — even for detections we end up dropping
    ingest_detection(detection, state["bands"], state["history"])

    label  = detection.get("label", "")
    result = {"detection": detection, "novelty_score": novelty}

    if label in PRIORITY_LABELS:
        result["reason"] = "priority_label"
        state["queue"].append(result)
        state["stats"]["queued"]   += 1
        state["stats"]["priority"] += 1
    elif novelty >= NOVELTY_THRESHOLD:
        result["reason"] = "novelty"
        state["queue"].append(result)
        state["stats"]["queued"] += 1
    else:
        result["reason"] = "dropped"
        state["stats"]["dropped"] += 1

    return result


def drain_queue(state):
    """Pop everything currently queued — call this when the 5G link is ready."""
    items = list(state["queue"])
    state["queue"].clear()
    return items


def get_stats(state):
    s = dict(state["stats"])
    s["queue_depth"] = len(state["queue"])
    total = s["seen"] or 1
    s["transmission_rate_pct"] = round(100 * s["queued"] / total, 1)
    return s


## DEMO

def make_detection(freq_hz, power_db, label, confidence=0.85, t=None):
    return {
        "timestamp":  t or time.time(),
        "freq_hz":    freq_hz,
        "bandwidth":  200_000,   # 200 kHz
        "power_db":   power_db,
        "label":      label,
        "confidence": confidence,
        "duration_s": 0.5,
    }


def run_demo():
    state = make_filter_state()
    t0 = 1_000_000.0

    print("── Transmission filter demo ─────────────────────────────────")
    print(f"{'t':>6}  {'freq MHz':>10}  {'power':>7}  {'label':20}  {'score':>6}  {'decision'}")
    print("─" * 75)

    example_events = [
        # Routine GSM traffic — establish baseline
        (0,  935.2e6, -72, "gsm_downlink"),
        (2,  935.2e6, -71, "gsm_downlink"),
        (4,  935.2e6, -73, "gsm_downlink"),
        (6,  935.2e6, -72, "gsm_downlink"),
        (8,  935.2e6, -71, "gsm_downlink"),
        # Routine WiFi — different freq, establish baseline
        (3,  2412e6,  -65, "wifi_beacon"),
        (5,  2412e6,  -66, "wifi_beacon"),
        (7,  2412e6,  -65, "wifi_beacon"),
        # Novel: unseen frequency band
        (10, 1090e6,  -55, "ads_b"),           # ADS-B, never seen before
        # Routine: GSM again (low novelty, baseline established)
        (11, 935.2e6, -72, "gsm_downlink"),
        # Novel: power spike on known freq
        (13, 935.2e6, -45, "gsm_downlink"),    # 27 dB above baseline — anomalous
        # Priority label regardless of novelty
        (15, 406e6,   -80, "emergency_beacon"),
        # Low confidence — dropped before scoring
        (17, 162.025e6, -60, "vhf_marine"),
        # Another routine event
        (20, 2412e6,  -65, "wifi_beacon"),
        # Novel: new label on seen freq
        (22, 2412e6,  -85, "unknown"),
        # Gap then return — recency score kicks in
        (28, 935.2e6, -72, "gsm_downlink"),
    ]

    low_conf = {6}  # index of the vhf_marine entry

    for i, (dt, freq, power, label) in enumerate(example_events):
        t    = t0 + dt
        conf = 0.3 if i in low_conf else 0.85
        det  = make_detection(freq, power, label, confidence=conf, t=t)
        res  = process_detection(det, state)
        score_str = f"{res['novelty_score']:.3f}"
        print(f"{dt:>6.0f}s  {freq/1e6:>10.3f}  {power:>5} dBm  {label:20}  {score_str:>6}  {res['reason']}")

    print()
    print("── Filter stats ─────────────────────────────────────────────")
    for k, v in get_stats(state).items():
        print(f"  {k:30} {v}")

    print()
    queued = drain_queue(state)
    print(f"── Queued for transmission ({len(queued)} detections) ────────────────")
    for r in queued:
        d = r["detection"]
        print(f"  {d['label']:22}  {d['freq_hz']/1e6:.3f} MHz  score={r['novelty_score']:.3f}  reason={r['reason']}")


if __name__ == "__main__":
    run_demo()