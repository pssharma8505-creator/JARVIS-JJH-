"""
core/audio_devices.py — pick which microphone and which speakers JARVIS uses.

WHY
    Both audio streams in main.py were opened without a `device=` argument, so
    they always took whatever the operating system called "default". On a laptop
    with a built-in mic, a webcam mic and a headset that is a coin toss — and on
    Windows the default *moves on its own* the moment you plug a headset in.
    "JARVIS can't hear me" almost always means "JARVIS is listening to the
    monitor's microphone".

WHY NAMES, NOT INDICES
    sounddevice identifies devices by integer index, and those indices shift
    whenever a device appears or disappears. Storing index 3 means that after
    unplugging a USB interface the saved setting silently points at something
    else. We store the device *name* and resolve it to an index at open time.

WHY THIS IS CACHED
    `sd.query_devices()` talks to the host audio API and can take a few hundred
    milliseconds on a Windows machine with many endpoints. Mark LV learned this
    lesson the expensive way — a 2.1-second `openwakeword` import on the Qt
    thread made the settings drawer look like it was broken. So the list is
    fetched once on a background thread at startup and served from cache.
"""

from __future__ import annotations

import re
import threading
import time

# The label shown for "let the OS decide", and the value stored in config for
# it. Empty string, so an untouched install and a deliberately-default install
# are the same thing — nothing changes for anyone who never opens the picker.
DEFAULT_LABEL = "System default"
DEFAULT_VALUE = ""

_cache: dict[str, list[str]] | None = None
_cache_lock = threading.Lock()

# PortAudio is not re-entrant for enumeration/probing, and reinit() tears the
# whole library down — everything that touches `sd` for discovery takes this.
_pa_lock = threading.RLock()

# Bumped every time the device list is rebuilt, so the UI can tell "my list is
# stale" from "nothing happened yet" without polling the OS itself.
_generation = 0

# Set when something happened that PortAudio cannot know about on its own
# (a Bluetooth headset connected, a USB mic was unplugged). The run loop calls
# consume_dirty() before it opens streams and, if set, re-enumerates first.
_dirty = False

# Which host API each direction settled on, so resolve() opens the same endpoint
# the picker listed. Filled in by _query().
_chosen_api: dict = {"input": None, "output": None}


# ── Why the raw list is unusable, and what is filtered out ───────────────────
#
# `sd.query_devices()` returns one entry per (device × host API), not one per
# device. Measured on a normal Windows machine: 41 entries for what the Windows
# sound settings show as 4 microphones and 4 speakers. The same Realtek
# microphone appears four times — once each under MME, DirectSound, WASAPI and
# WDM-KS — and none of the four is labelled to say which is which.
#
# Handing that to a person is not a choice, it is a quiz. So the list is reduced
# the way the operating system's own settings panel does it:
#
#   1. ONE host API per direction — chosen by measurement, not by reasoning.
#      See the preference note below.
#   2. No pseudo-devices. "Microsoft Sound Mapper", "Primary Sound Driver",
#      ALSA's "default"/"sysdefault"/"dmix" are aliases for "whatever the OS
#      picks" — which is precisely the "System default" entry already at the top
#      of the list. Offering them again as if they were hardware is noise.
#   3. No zero-channel or unnamed entries (WASAPI reports one of each).
#   4. Deduplicated by name.
#
# Nothing is hidden that a person could actually want: the same hardware is
# still there, listed once, under the name their operating system uses for it.

# ── Preference order, which is only a starting point ─────────────────────────
#
# Two earlier versions of this file were wrong in the same way: they decided
# which host API to use by reasoning about it instead of measuring it.
#
#   v1 ranked APIs by how clean their device names were and picked WASAPI.
#      WASAPI in shared mode does not resample — the hardware runs at 48 kHz,
#      this app streams 16 kHz in and 24 kHz out, and every open failed with
#      "Invalid sample rate". The picker looked right and did nothing.
#
#   v2 added a rate check and picked DirectSound, which passes that check on
#      both sides. PortAudio's DirectSound *output* is a silent sink: the
#      stream opens, every write returns success in ~0 ms, and nothing is ever
#      heard. Same failure, one layer deeper.
#
# So this list is a preference, not a promise. Which API actually gets used is
# decided below by _usable() (open it for real) and _transport_works() (does
# audio actually move), per direction. On this platform that lands on
# DirectSound for the microphone and MME for the speakers — a split that no
# amount of reasoning would have produced.
_PREFERRED_APIS = {
    "Windows": ("directsound", "mme", "wasapi"),
    # macOS has only Core Audio, so there is nothing to disambiguate.
    "Darwin":  ("core audio",),
    # PulseAudio/PipeWire present one clean endpoint per device; raw ALSA
    # presents dozens of routing permutations of the same card.
    "Linux":   ("pulse", "pipewire", "jack", "alsa"),
}

# ── "It opens" is not "it works" ─────────────────────────────────────────────
#
# Opening a stream successfully proves nothing. Measured, writing 2.0 s of audio:
#
#     device=None (MME)                    2.02 s   consumed in real time
#     DirectSound, any output device       0.00 s   swallowed instantly
#
# No flag or capability field reports this. The only thing that separates a real
# sink from a fake one is whether it consumes audio at the rate audio is
# consumed at — so that is what gets measured, once per host API per direction,
# on the background thread at startup, using silence.
#
# Each direction is probed **the way main.py actually uses it**. That is not a
# detail: DirectSound input passes a callback stream and fails a blocking read,
# so an earlier version of this probe rejected a microphone that works perfectly
# in the app. Probe the mode you ship, not the mode that is easier to write.
_PROBE_SECONDS = {"output": 0.6, "input": 0.35}

# Cache: {(api_name_or_None, kind): bool}
_probe_results: dict = {}


# ── Which format will this device really open at? ────────────────────────────
#
# main.py streams the microphone at 16 kHz and plays the model at 24 kHz, and the
# first versions of this module treated a device that would not open at exactly
# those numbers as "not usable". That is the wrong rule for Bluetooth. A headset
# in hands-free (HFP) mode runs at 8 or 16 kHz, a stereo (A2DP) sink at 44.1 or
# 48 kHz, WASAPI will not resample, and some endpoints only accept two channels.
# Such devices were silently dropped from the picker — or, if saved, silently
# swapped for the system default — which is the "earbuds never work" bug.
#
# So instead of asking "does it open at OUR format?", find a format it DOES open
# at and let the app convert (to_mono_16k / from_mono, below).
_EXTRA_RATES = (48000, 44100, 16000, 8000, 32000, 24000, 22050)

# {(device_index, kind): (rate, channels) | None}. Cleared by reinit()/configure().
_params_cache: dict = {}


def _try_open(idx: int, kind: str, rate: int, ch: int) -> bool:
    """Open a real stream at this format and close it again."""
    st = None
    try:
        import sounddevice as sd
        if kind == "input":
            st = sd.InputStream(samplerate=rate, channels=ch, dtype="int16",
                                blocksize=1024, device=idx,
                                callback=lambda *_a: None)
        else:
            st = sd.RawOutputStream(samplerate=rate, channels=ch, dtype="int16",
                                    blocksize=1024, device=idx)
        st.start()
        return True
    except Exception:
        return False
    finally:
        if st is not None:
            try:
                st.stop(); st.close()
            except Exception:
                pass


def _probe_params(idx: int, kind: str):
    """(rate, channels) this device opens at, preferring the app's own rate and
    mono, or None if it opens at nothing. Cached per device."""
    key = (idx, kind)
    with _pa_lock:
        if key in _params_cache:
            return _params_cache[key]

        rates = [_RATES.get(kind, 16000)]
        max_ch = 2
        try:
            import sounddevice as sd
            info = sd.query_devices(idx)
            native = int(round(info.get("default_samplerate") or 0))
            if native:
                rates.append(native)
            chan_key = "max_input_channels" if kind == "input" else "max_output_channels"
            max_ch = int(info.get(chan_key, 2) or 2)
        except Exception:
            pass
        rates += list(_EXTRA_RATES)
        rates = list(dict.fromkeys(r for r in rates if r > 0))

        result = None
        for ch in (1, 2):
            if ch > max(max_ch, 1):
                continue
            for rate in rates:
                if _try_open(idx, kind, rate, ch):
                    result = (rate, ch)
                    break
            if result:
                break
        _params_cache[key] = result
        return result


def open_params(dev, kind: str) -> tuple[int, int]:
    """(rate, channels) to open `dev` with — `dev` being what resolve() returned,
    so None means the system default. Never raises; if nothing can be probed it
    returns the app's own format and lets the real open report the real error."""
    fallback = (_RATES.get(kind, 16000), 1)
    try:
        import sounddevice as sd
        idx = dev
        if idx is None:
            try:
                idx = sd.default.device[0 if kind == "input" else 1]
            except Exception:
                idx = None
            if idx is None or idx < 0:
                try:
                    idx = sd.query_devices(kind=kind).get("index")
                except Exception:
                    idx = None
        if idx is None or idx < 0:
            return fallback
        return _probe_params(int(idx), kind) or fallback
    except Exception:
        return fallback


def to_mono_16k(block, src_rate: int, dst_rate: int):
    """Microphone block (n, channels) int16 at src_rate -> (m, 1) int16 at dst_rate."""
    import numpy as np
    a = np.asarray(block)
    a = a.mean(axis=1) if (a.ndim == 2 and a.shape[1] > 1) else a.reshape(-1)
    a = a.astype(np.float32)
    if src_rate != dst_rate and a.size:
        n_out = int(round(a.size * dst_rate / src_rate))
        if n_out <= 0:
            return np.zeros((0, 1), dtype=np.int16)
        if src_rate > dst_rate:
            k = int(round(src_rate / dst_rate))
            if k > 1:   # crude low-pass so the decimation does not alias
                a = np.convolve(a, np.ones(k, dtype=np.float32) / k, mode="same")
        a = np.interp(np.linspace(0, a.size - 1, n_out), np.arange(a.size), a)
    return np.clip(a, -32768, 32767).astype(np.int16).reshape(-1, 1)


def from_mono(raw: bytes, src_rate: int, dst_rate: int, channels: int) -> bytes:
    """Mono int16 PCM bytes at src_rate -> PCM bytes at dst_rate with `channels`."""
    if src_rate == dst_rate and channels == 1:
        return raw
    import numpy as np
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if src_rate != dst_rate and a.size:
        n_out = int(round(a.size * dst_rate / src_rate))
        if n_out <= 0:
            return b""
        if src_rate > dst_rate:
            k = int(round(src_rate / dst_rate))
            if k > 1:
                a = np.convolve(a, np.ones(k, dtype=np.float32) / k, mode="same")
        a = np.interp(np.linspace(0, a.size - 1, n_out), np.arange(a.size), a)
    a = np.clip(a, -32768, 32767).astype(np.int16)
    if channels > 1:
        a = np.repeat(a[:, None], channels, axis=1)
    return a.tobytes()


def _transport_works(idx: int, kind: str) -> bool:
    """Does this ONE device actually move audio, or only pretend to?

    Opened at whatever format _probe_params() found for it. Output writes
    silence, so the probe is inaudible; input reads and discards."""
    params = _probe_params(idx, kind)
    if params is None:
        return False
    rate, ch = params
    secs = _PROBE_SECONDS.get(kind, 0.5)
    try:
        import sounddevice as sd
        # Each direction is probed the way main.py actually uses it. That is not
        # a detail: DirectSound input passes a callback stream and fails a
        # blocking read, so probing the wrong mode rejected a microphone that
        # works perfectly in the app.
        if kind == "output":
            # main.py writes with stream.write() — a real sink is rate-limited
            # by the hardware clock, a fake one swallows the buffer instantly.
            st = sd.RawOutputStream(samplerate=rate, channels=ch, dtype="int16",
                                    blocksize=1024, device=idx)
            st.start()
            t0 = time.monotonic()
            st.write(bytes(int(rate * secs) * 2 * ch))   # silence — inaudible
            elapsed = time.monotonic() - t0
            st.stop(); st.close()
            return elapsed > secs * 0.5
        # main.py reads through a callback — count what arrives.
        frames = [0]

        def _cb(indata, n, *_a):
            frames[0] += n

        st = sd.InputStream(samplerate=rate, channels=ch, dtype="int16",
                            blocksize=1024, device=idx, callback=_cb)
        st.start()
        time.sleep(secs)
        st.stop(); st.close()
        return frames[0] > rate * secs * 0.3
    except Exception as e:
        print(f"[Audio] {kind} transport probe failed on device {idx}: {e}")
        return False


def _api_moves_audio(found, kind: str, api_key) -> bool:
    """Does this host API move audio at all? Judged on up to three of its
    devices — NOT just the first. A disconnected or asleep Bluetooth endpoint
    listed first used to condemn an API whose other devices were fine."""
    if api_key in _probe_results:
        return _probe_results[api_key]
    ok = False
    for idx, _name in found[:3]:
        if _transport_works(idx, kind):
            ok = True
            break
    if not ok:
        print(f"[Audio] {kind}: host API {api_key[0] or 'any'} moves no audio "
              f"on its first {min(len(found), 3)} device(s) — skipping it")
    _probe_results[api_key] = ok
    return ok


def _display_name(name: str, devices) -> str:
    """MME truncates device names to 31 characters, so the API that actually
    carries the audio may not be the one that can spell. If another host API
    knows a longer name that starts with this one, show that instead — the user
    reads 'Realtek HD Audio 2nd output (Realtek(R) Audio)' while the stream runs
    on the endpoint called 'Realtek HD Audio 2nd output (Re'."""
    if len(name) < 30:
        return name
    best = name
    for dev in devices:
        other = (dev.get("name") or "").strip()
        if len(other) > len(best) and other.startswith(name):
            best = other
    return best

# The rates the app opens its streams at. Defaults match main.py; main.py calls
# configure() at startup with its own constants so the two can never drift apart
# and silently reintroduce the bug above.
_RATES = {"input": 16000, "output": 24000}


def configure(input_rate: int, output_rate: int) -> None:
    """Tell this module the sample rates the audio streams will use, so the
    picker can rule out devices that cannot be opened at them.

    Drops any cached list: which devices are usable depends on the rate, so a
    list built under the old rates would be stale."""
    global _cache
    _RATES["input"]  = int(input_rate)
    _RATES["output"] = int(output_rate)
    _params_cache.clear()
    _probe_results.clear()
    with _cache_lock:
        _cache = None


def _usable(idx: int, kind: str) -> bool:
    """Can this device be opened in SOME format we can convert from/to?

    Deliberately opens a real stream rather than asking
    `check_output_settings`, because that function lies: it passed for an MME
    endpoint that then failed to open with "The specified format is not
    supported or cannot be translated" [MME error 32]. Opening and immediately
    closing costs milliseconds and is the only answer that holds."""
    return _probe_params(idx, kind) is not None


# Aliases for "the default device" and internal routing endpoints. Matched
# case-insensitively as substrings against the device name.
_PSEUDO_DEVICES = (
    "sound mapper",        # Windows MME
    "primary sound",       # Windows DirectSound ("Primary Sound Capture Driver")
    "sysdefault",          # ALSA
    "default",             # ALSA / PulseAudio alias
    "dmix", "dsnoop",      # ALSA software mixing plugins
    "surround",            # ALSA channel-layout permutations of one card
    "samplerate", "speexrate", "upmix", "vdownmix", "null",
)


def _is_pseudo(name: str) -> bool:
    low = name.lower()
    return any(tok in low for tok in _PSEUDO_DEVICES)


def _query() -> dict[str, list[str]]:
    """Return {'input': [names...], 'output': [names...]}. Never raises.

    Only real, selectable devices — see the note above."""
    out: dict[str, list[str]] = {"input": [], "output": []}
    _pa_lock.acquire()
    try:
        import platform
        import sounddevice as sd

        devices = list(sd.query_devices())
        try:
            apis = [a.get("name", "") for a in sd.query_hostapis()]
        except Exception:
            apis = []

        preferred = _PREFERRED_APIS.get(platform.system(), ())

        def _collect(api_filter, kind) -> list[tuple[int, str]]:
            """(index, name) for named, non-pseudo devices on one side that can
            be opened at the rate that side runs at."""
            chan = "max_input_channels" if kind == "input" else "max_output_channels"
            found, seen = [], set()
            for idx, dev in enumerate(devices):
                name = (dev.get("name") or "").strip()
                if not name or _is_pseudo(name) or name in seen:
                    continue
                if dev.get(chan, 0) <= 0:
                    continue
                if api_filter is not None:
                    api = apis[dev["hostapi"]].lower() if dev.get("hostapi", -1) < len(apis) else ""
                    if api_filter not in api:
                        continue
                if not _usable(idx, kind):
                    continue
                seen.add(name)
                found.append((idx, name))
            return found

        # Each direction picks its own host API. They are genuinely different
        # problems — on Windows the microphone works on DirectSound while the
        # speakers only work on MME — and a single global choice cannot be right
        # for both.
        for kind in ("input", "output"):
            for api_filter in list(preferred) + [None]:
                found = _collect(api_filter, kind)
                if not found:
                    continue
                # One verdict per API per direction, cached, on this thread.
                if not _api_moves_audio(found, kind, (api_filter, kind)):
                    continue
                _chosen_api[kind] = api_filter
                out[kind] = [_display_name(n, devices) for _i, n in found]
                break
            if out[kind]:
                print(f"[Audio] {kind}: using "
                      f"{_chosen_api[kind] or 'any host API'} "
                      f"({len(out[kind])} devices)")
        return out

    except Exception as e:
        print(f"[Audio] Device enumeration failed: {e}")
    finally:
        _pa_lock.release()
    return out


def prefetch() -> None:
    """Warm the cache on a background thread. Called once at startup so the
    settings drawer never pays for enumeration on the Qt thread."""
    def _work():
        global _cache, _generation
        result = _query()
        with _cache_lock:
            _cache = result
            _generation += 1
        print(f"[Audio] {len(result['input'])} input / "
              f"{len(result['output'])} output devices found")
    threading.Thread(target=_work, daemon=True, name="audio-devices").start()


def list_devices(kind: str, refresh: bool = False) -> list[str]:
    """Device names for 'input' or 'output'. Falls back to a synchronous query
    if the prefetch has not landed yet — correctness over the cache."""
    global _cache
    with _cache_lock:
        cached = None if refresh else _cache
    if cached is None:
        cached = _query()
        with _cache_lock:
            _cache = cached
    return list(cached.get(kind, []))


def generation() -> int:
    """Increments each time the device list is rebuilt."""
    return _generation


def mark_dirty() -> None:
    """Something changed that PortAudio cannot see by itself (hotplug, Bluetooth
    connect). The next consume_dirty() returns True exactly once."""
    global _dirty
    _dirty = True


def consume_dirty() -> bool:
    global _dirty
    was, _dirty = _dirty, False
    return was


def reinit() -> None:
    """Re-enumerate the operating system's audio devices from scratch.

    PortAudio snapshots the device list when it is initialised and never looks
    again — a Bluetooth headset connected after launch is invisible to it until
    the library is torn down and brought back up. That is only legal while NO
    stream is open, so the run loop calls this between sessions, after the old
    streams are closed and before the new ones open. Never raises."""
    global _cache, _generation
    with _pa_lock:
        try:
            import sounddevice as sd
            sd._terminate()
            sd._initialize()
        except Exception as e:
            print(f"[Audio] PortAudio re-initialise failed: {e}")
        _params_cache.clear()
        _probe_results.clear()
        _chosen_api["input"] = _chosen_api["output"] = None
        result = _query()
        with _cache_lock:
            _cache = result
            _generation += 1
        print(f"[Audio] Re-scanned: {len(result['input'])} input / "
              f"{len(result['output'])} output devices")


# ── Bluetooth: one headset, two endpoints ────────────────────────────────────
#
# Windows exposes a Bluetooth headset as TWO playback devices:
#     "Headphones (X Stereo)"            A2DP — high quality, NO microphone
#     "Headset (X Hands-Free AG Audio)"  HFP  — 8/16 kHz, has the microphone
# and one capture device: "Headset (X Hands-Free AG Audio)". The instant a
# program opens that microphone the headset switches profile, the stereo
# endpoint goes away, and audio sent to it is lost. Using the earbuds' mic with
# the earbuds' *stereo* output therefore cannot work — both directions have to
# sit on the Hands-Free pair.
_BT_RE = re.compile(r"\((.+?)\s+(hands-free(?:\s+ag\s+audio)?|ag\s+audio|stereo)\)\s*$",
                    re.IGNORECASE)


def _bt_parts(name: str):
    """(device_base_name_lowercase, is_hands_free) or None if not a BT endpoint."""
    m = _BT_RE.search(name or "")
    if not m:
        return None
    kind = m.group(2).lower()
    return m.group(1).strip().lower(), ("hands-free" in kind or "ag" in kind)


def _default_name(kind: str) -> str:
    try:
        import sounddevice as sd
        name = (sd.query_devices(kind=kind).get("name") or "").strip()
    except Exception:
        return ""
    # MME truncates to 31 characters; expand it from the picker's list.
    for n in list_devices(kind):
        if name and n.startswith(name[:24]):
            return n
    return name


def reconcile_bluetooth(in_name: str, out_name: str) -> str:
    """Return the speaker name to use, moving it onto the matching Hands-Free
    endpoint when the microphone is a Bluetooth headset's. Used for this run
    only — the saved setting is left alone."""
    try:
        in_eff = (in_name or "").strip()
        if not in_eff or in_eff == DEFAULT_LABEL:
            in_eff = _default_name("input")
        mic = _bt_parts(in_eff)
        if not mic or not mic[1]:
            return out_name
        base = mic[0]
        hands_free = [n for n in list_devices("output")
                      if (p := _bt_parts(n)) and p[0] == base and p[1]]
        if not hands_free:
            return out_name

        out_eff = (out_name or "").strip()
        if not out_eff or out_eff == DEFAULT_LABEL:
            # "System default" only needs moving if the default IS this headset.
            if base[:18] in _default_name("output").lower():
                print(f"[Audio] Bluetooth mic in use — routing speaker to "
                      f"'{hands_free[0]}' (the stereo endpoint drops out when the "
                      f"mic opens)")
                return hands_free[0]
            return out_name
        cur = _bt_parts(out_eff)
        if cur and cur[0] == base and not cur[1]:
            print(f"[Audio] Bluetooth mic in use — routing speaker to "
                  f"'{hands_free[0]}' instead of the stereo endpoint")
            return hands_free[0]
    except Exception as e:
        print(f"[Audio] Bluetooth routing check failed: {e}")
    return out_name


def resolve(name: str, kind: str):
    """Turn a saved device name into something sounddevice accepts.

    Returns None for "system default" — which is also what we return when the
    saved device is gone, because a missing headset must degrade to the built-in
    speakers, not to a crash on startup.

    Candidates are walked in the same host-API order the picker used, so a name
    the user chose from the WASAPI list resolves to the WASAPI endpoint. Without
    that ordering a full name would fall through to MME's truncated copy of the
    same device — which happens to work, but means the setting quietly refers to
    a different endpoint than the one on screen."""
    wanted = (name or "").strip()
    if not wanted or wanted == DEFAULT_LABEL:
        return None

    try:
        import platform
        import sounddevice as sd

        devices = list(sd.query_devices())
        try:
            apis = [a.get("name", "") for a in sd.query_hostapis()]
        except Exception:
            apis = []

        want_in  = (kind == "input")
        chan_key = "max_input_channels" if want_in else "max_output_channels"

        def _candidates(api_filter):
            for idx, dev in enumerate(devices):
                if dev.get(chan_key, 0) <= 0:
                    continue
                if api_filter is not None:
                    api = apis[dev["hostapi"]].lower() if dev.get("hostapi", -1) < len(apis) else ""
                    if api_filter not in api:
                        continue
                yield idx, (dev.get("name") or "").strip()

        # The API the picker settled on for this direction comes first — the
        # endpoint that was listed must be the endpoint that gets opened, or the
        # setting means something different from what it says. list_devices()
        # populates it; calling it here is a no-op once the cache is warm.
        list_devices(kind)
        chosen = _chosen_api.get(kind)
        orders = ([chosen] if chosen is not None else []) \
            + [a for a in _PREFERRED_APIS.get(platform.system(), ()) if a != chosen] \
            + [None]

        # A candidate only counts if it can be opened at the rate this side runs
        # at. The prefix match matters because the API that carries the audio is
        # not always the one that can spell: MME truncates names to 31 characters
        # while DirectSound and WASAPI do not, so the name shown in the picker
        # can be longer than the name of the endpoint it actually opens.
        for api_filter in orders:
            partial = None
            for idx, dev_name in _candidates(api_filter):
                if dev_name == wanted:
                    if _usable(idx, kind):
                        return idx
                    continue
                if partial is None and (dev_name.startswith(wanted[:24])
                                        or wanted.startswith(dev_name[:24])):
                    if _usable(idx, kind):
                        partial = idx
            if partial is not None:
                return partial

        print(f"[Audio] Saved {kind} device '{wanted}' cannot be opened in any "
              f"format on any host API — using system default")
        return None
    except Exception as e:
        print(f"[Audio] resolve({kind}) failed: {e} — using system default")
        return None
