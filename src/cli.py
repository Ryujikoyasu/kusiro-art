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
def show(simulate: bool = typer.Option(True, "--simulate/--no-simulate")):
    """Show the U-shape layout and simulate wave."""
    cfg = load_config()
    idx = build_u_shape_idx(cfg)
    vp = Viewport(cfg, idx)
    vp.run_simulation()


@app.command()
def run():
    """Run real control (camera trigger + serial + audio)."""
    from .main_real import run as run_real

    run_real()


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
    """Show insect color base/accent and sound files from structure."""
    try:
        from config.config_structure import get_insect_base_config  # type: ignore
    except Exception as e:
        raise typer.BadParameter(f"Failed to import insect structure: {e}")
    data = get_insect_base_config()
    for key, v in data.items():
        colors = v.get("colors", {})
        sounds = v.get("sound_files", {})
        typer.echo(f"{key}: base={colors.get('base')} accent={colors.get('accent')} files={sounds}")


def main():  # python -m src.cli
    app()


if __name__ == "__main__":
    main()
