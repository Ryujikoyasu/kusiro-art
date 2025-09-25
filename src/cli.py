import json
import time
import typer
from pathlib import Path

from .led.mapper import build_u_shape_idx
from .led.serial_link import SerialLink
from .audio.engine import AudioEngine
from .sim.viewport import Viewport
from .util.config import load_config, save_config


app = typer.Typer(add_completion=False, help="Kushiro insect system CLI")
colors_app = typer.Typer(help="Color utilities")
app.add_typer(colors_app, name="colors")


@app.command()
def layout_export(output: Path = typer.Option(Path("assets/layouts/u_shape_95x10.json"), "--output")):
    """Export U-shape LED layout mapping JSON."""
    cfg = load_config()
    idx = build_u_shape_idx(cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)
    typer.echo(f"Exported layout to {output}")


@app.command()
def serial_ping():
    """Check Arduino R4 serial connectivity."""
    cfg = load_config()
    link = SerialLink(cfg)
    with link:
        pong = link.ping()
        typer.echo(f"PING -> {pong}")


@app.command()
def audio_check():
    """List insect sounds and target gains/pans."""
    cfg = load_config()
    engine = AudioEngine(cfg)
    engine.print_assets()


@app.command("audio_sync")
def audio_sync():
    """Analyze insect sounds and write config/insect_config.py (INSECT_PARAMS)."""
    # Import and run generator in-process to avoid subprocess complexity
    try:
        from audio_sync_generator import generate_insect_config  # type: ignore
    except Exception as e:  # pragma: no cover
        raise typer.BadParameter(f"Failed to import audio_sync_generator: {e}")
    generate_insect_config()
    typer.echo("Generated config/insect_config.py")


@app.command("audio_normalize")
def audio_normalize(
    directory: str = typer.Option("assets/data/sound/trimmed", "--dir", help="Directory containing mp3s"),
    backup: str = typer.Option("assets/data/sound/backup_originals", "--backup", help="Backup directory for originals"),
    lufs: float = typer.Option(-16.0, help="Target integrated loudness (LUFS) e.g. -16"),
    tp: float = typer.Option(-1.0, help="True peak target (dBFS) e.g. -1.0"),
    lra: float = typer.Option(11.0, help="Loudness range target (LU)"),
    bitrate: str = typer.Option("192k", help="Output bitrate for mp3 re-encode"),
):
    """Normalize loudness of mp3s in a directory. Backs up originals, overwrites in place."""
    try:
        from .tools.normalize_audio import normalize_dir  # type: ignore
    except Exception as e:
        raise typer.BadParameter(f"Could not import normalizer: {e}")
    normalize_dir(Path(directory), Path(backup), I=lufs, TP=tp, LRA=lra, bitrate=bitrate)
    typer.echo(f"Normalized mp3s in {directory}. Backups in {backup}.")


@app.command()
def kakon_watch():
    """Run camera-based kakon detector and send serial wave events."""
    from .detect.kakon_cam import KakomCameraDetector

    cfg = load_config()
    idx = build_u_shape_idx(cfg)
    link = SerialLink(cfg)
    det = KakomCameraDetector(cfg)
    with link:
        link.send_conf(total=len(idx))
        typer.echo("Watching for kakon... Press Ctrl+C to stop")
        try:
            for _ in det.watch():
                typer.echo("Trigger detected -> sending wave")
                _run_wave_once(cfg, link)
        except KeyboardInterrupt:
            pass


def _run_wave_once(cfg, link: SerialLink):
    speed = float(cfg["wave"]["speed_mps"])  # m/s
    total_m = sum(cfg["segments_m"].values())
    duration = total_m / speed
    t0 = time.time()
    while True:
        t = time.time() - t0
        pos = min(1.0, t / duration)
        link.send_wave(pos=pos, tail_m=float(cfg["wave"]["tail_m"]), bright=200)
        if pos >= 1.0:
            break
        time.sleep(1 / 60.0)


@app.command()
def show(
    simulate: bool = typer.Option(True, "--simulate/--no-simulate"),
    version: int = typer.Option(None, "--version", help="Effect version: 1=orange wave, 2=calm blue"),
    mirror: bool = typer.Option(False, "--mirror", help="Mirror frames to device via MAGIC_BYTE"),
):
    """Show the U-shape layout and simulate wave (version 1 or 2)."""
    cfg = load_config()
    if version is not None:
        cfg.setdefault("sim", {})
        cfg["sim"]["effect_version"] = int(version)
    idx = build_u_shape_idx(cfg)
    vp = Viewport(cfg, idx, mirror_to_device=mirror)
    vp.run_simulation()


@app.command()
def run(version: int = typer.Option(None, "--version", help="Effect version: 1=orange wave, 2=calm blue")):
    """Run real control (auto kakon + serial + audio), versioned effect."""
    from .main_real import run as run_real
    run_real(effect_version=version)


@app.command()
def trigger_once():
    """Send a single wave immediately to the device (manual trigger)."""
    cfg = load_config()
    idx = build_u_shape_idx(cfg)
    link = SerialLink(cfg)
    total = len(idx)
    speed = float(cfg["wave"]["speed_mps"])  # m/s
    total_m = sum(cfg["segments_m"].values())
    duration = total_m / max(1e-6, speed)
    tail_m = float(cfg["wave"]["tail_m"])  # m
    with link:
        link.send_conf(total=total)
        t0 = time.time()
        while True:
            t = time.time() - t0
            pos = min(1.0, t / duration)
            link.send_wave(pos=pos, tail_m=tail_m, bright=220)
            if pos >= 1.0:
                break
            time.sleep(1 / 60.0)
    typer.echo("Wave completed.")


@app.command()
def black():
    """Turn all LEDs off on the device."""
    cfg = load_config()
    link = SerialLink(cfg)
    with link:
        link.black()
    typer.echo("BLACK sent.")


@colors_app.command("pick")
def colors_pick():
    """Open the color wheel picker and send color to device (MAGIC_BYTE)."""
    # Dynamically import root script
    import importlib.util, sys, pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    path = root / "color_wheel_picker.py"
    spec = importlib.util.spec_from_file_location("color_wheel_picker", str(path))
    if not spec or not spec.loader:
        raise typer.BadParameter("Could not load color_wheel_picker.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["color_wheel_picker"] = mod
    spec.loader.exec_module(mod)  # type: ignore
    if hasattr(mod, "interactive_color_picker"):
        mod.interactive_color_picker()
    else:
        raise typer.BadParameter("interactive_color_picker() not found")


@colors_app.command("insect")
def colors_insect():
    """Show insect names and base/accent colors, send to device (MAGIC_BYTE)."""
    from .tools.insect_color_tester import run as run_tester
    run_tester()


@app.command()
def config_set(key: str, value: str):
    """Set a config key to a value (dot notation supported)."""
    cfg = load_config()
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    # Try to parse JSON literal
    try:
        cur[parts[-1]] = json.loads(value)
    except Exception:
        cur[parts[-1]] = value
    save_config(cfg)
    typer.echo(f"Set {key} = {cur[parts[-1]]}")


@app.command("config_insects")
def config_insects():
    """Show insect color and sound files from structure."""
    try:
        from config.config_structure import get_insect_base_config  # type: ignore
    except Exception as e:
        raise typer.BadParameter(f"Failed to import insect structure: {e}")
    data = get_insect_base_config()
    for key, v in data.items():
        colors = v.get("colors", {})
        sounds = v.get("sound_files", {})
        col = colors.get('color') or colors.get('base')
        typer.echo(f"{key}: color={col} files={sounds}")


def main():  # python -m src.cli
    app()


if __name__ == "__main__":
    main()
