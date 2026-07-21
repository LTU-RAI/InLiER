# Standard figlet "InLiER" (kept as plain ASCII).
_LOGO = r"""
  ___       _     _ _____ ____
 |_ _|_ __ | |   (_) ____|  _ \
  | || '_ \| |   | |  _| | |_) |
  | || | | | |___| | |___|  _ <
 |___|_| |_|_____|_|_____|_| \_\
"""

_WIDTH = 50


def print_banner(version: str = "1.0.0"):
    print(_LOGO.rstrip("\n"))
    print()
    print(" Intermediate LiDAR Encoding for Retrieval")
    print()
    print(" (c) 2026, Luleå University of Technology, Sweden")
    print(" version : " + str(version))
    print("=" * _WIDTH)


def print_config_banner(config):
    """Print selected encoder parameters as a plain aligned table."""
    rows = [
        ("N_h", getattr(config, "N_h", None)),
        ("z_min", getattr(config, "z_min", None)),
        ("z_max", getattr(config, "z_max", None)),
        ("r_max", getattr(config, "r_max", None)),
        ("N_r", getattr(config, "N_r", None)),
        ("N_a", getattr(config, "N_a", None)),
        ("N_s", getattr(config, "N_s", None)),
        ("cell_size", getattr(config, "cell_size", None)),
        ("xy_max", getattr(config, "xy_max", None)),
        ("window", getattr(config, "window", None)),
        ("max_kp_per_slice", getattr(config, "max_kp_per_slice", None)),
        ("max_kp_total", getattr(config, "max_kp_total", None)),
    ]

    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.1f}"
        return str(v)

    print("Encoder configuration")
    print("-" * _WIDTH)
    for key, value in rows:
        if value is None:
            continue
        disp = value
        if key == "max_kp_total" and isinstance(value, (int, float)):
            disp = int(round(float(value)))
        print(f"  {key.ljust(18)}: {_fmt(disp)}")

    # Derived per-bin resolutions.
    try:
        n_h = int(getattr(config, "N_h", 0))
        n_r = int(getattr(config, "N_r", 0))
        n_a = int(getattr(config, "N_a", 0))
        n_s = int(getattr(config, "N_s", 0))
        z_min = float(getattr(config, "z_min", 0.0))
        z_max = float(getattr(config, "z_max", 0.0))
        r_max = float(getattr(config, "r_max", 0.0))

        h_res = (z_max - z_min) / n_h if n_h > 0 else float("nan")
        r_res = r_max / n_r if n_r > 0 else float("nan")
        a_res = 360.0 / n_a if n_a > 0 else float("nan")

        # Matches the encoder logic: <=3 -> 1/1/1, <=5 -> 2/2/1, else -> 3/3/1.
        if n_s <= 3:
            n_lin, n_plan, n_scatter = 1, 1, 1
        elif n_s <= 5:
            n_lin, n_plan, n_scatter = 2, 2, 1
        else:
            n_lin, n_plan, n_scatter = 3, 3, 1

        print("-" * _WIDTH)
        print("Derived resolutions")
        print("-" * _WIDTH)
        print(f"  {'height'.ljust(18)}: {h_res:.1f} m/bin")
        print(f"  {'radial'.ljust(18)}: {r_res:.1f} m/bin")
        print(f"  {'azimuth'.ljust(18)}: {a_res:.1f} deg/bin")
        print(f"  {'L/P/S'.ljust(18)}: {n_lin}/{n_plan}/{n_scatter} bin(s)")
    except Exception:
        # Keep startup robust even if config fields are missing/unexpected.
        pass

    print("=" * _WIDTH)
    print()
