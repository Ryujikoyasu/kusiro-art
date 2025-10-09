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


    #
    # Detection-related commands removed.


@app.command()
def black():
    """Turn all LEDs off on the device."""
    cfg = load_config()
    link = SerialLink(cfg)
    with link:
        link.black()
    typer.echo("BLACK sent.")


@app.command()
def ambient(
    species_count: int = typer.Option(3, "--species", help="Number of species active at once (default 3)"),
    change_interval: float = typer.Option(60.0, "--change-interval", help="Seconds between species rotation"),
    wave_seconds: float = typer.Option(20.0, "--wave-seconds", help="Seconds per volume wave cycle"),
    density: float = typer.Option(1.0, "--density", min=0.1, max=5.0, help=">1: more individuals and chirps; <1: fewer"),
    mirror: bool = typer.Option(False, "--mirror", help="Mirror frames to device via MAGIC_BYTE"),
):
    """Ambient simulation: 10s volume waves, rotate species every 60s, with window and optional mirroring."""
    cfg = load_config()
    idx = build_u_shape_idx(cfg)
    vp = Viewport(cfg, idx, mirror_to_device=mirror)
    # Scale concurrency and intervals with density
    density = max(0.1, float(density))
    # Base from config
    simcfg = cfg.get("sim", {})
    base_total = int(simcfg.get("max_concurrent_total", simcfg.get("max_concurrent_chirps", 24)))
    base_per = int(simcfg.get("max_concurrent_per_species", 12))
    max_total_override = max(1, int(round(base_total * density)))
    max_per_species_override = max(1, int(round(base_per * density)))
    interval_scale = 1.0 / density  # more density -> shorter intervals
    vp.run_ambient_simulation(species_count=max(1, int(species_count)),
                              change_interval=max(1.0, float(change_interval)),
                              wave_seconds=max(0.5, float(wave_seconds)),
                              max_total_override=max_total_override,
                              max_per_species_override=max_per_species_override,
                              interval_scale=interval_scale)


@app.command("ambient_record")
def ambient_record(
    out: Path = typer.Option(Path("recordings/ambient.wav"), "--out", help="Output WAV path"),
    seconds: float = typer.Option(60.0, "--seconds", help="Record duration (sec)"),
    species_count: int = typer.Option(3, "--species", help="Number of species active at once (default 3)"),
    change_interval: float = typer.Option(60.0, "--change-interval", help="Seconds between species rotation"),
    wave_seconds: float = typer.Option(20.0, "--wave-seconds", help="Seconds per volume wave cycle"),
    density: float = typer.Option(1.0, "--density", min=0.1, max=5.0, help=">1: more individuals and chirps; <1: fewer"),
):
    """Record ambient insect audio to a WAV file (headless)."""
    from .tools.ambient_recorder import run_record
    typer.echo(f"Recording ambient for {int(seconds)}s to {out} ...")
    path = run_record(out_path=out,
                      seconds=max(1.0, float(seconds)),
                      species_count=max(1, int(species_count)),
                      change_interval_s=max(1.0, float(change_interval)),
                      wave_seconds=max(0.5, float(wave_seconds)),
                      density=max(0.1, float(density)))
    typer.echo(f"Recorded to {path}")


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


    # Detection tuner removed.


def main():  # python -m src.cli
    app()


if __name__ == "__main__":
    main()
