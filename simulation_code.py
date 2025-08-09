import argparse, time, math, json
from pathlib import Path
import numpy as np
import pandas as pd
import meep as mp

light_speed = 299792458.0

def ensure_integer_grid(s, res):
    px = round(s * res)
    if px < 4:
        px = 4
    return px / res, px

def run_time_series(f, geom, sx, sy, res, probe_point, sample_oversample=30, n_cycles=80, cour=0.5):
    f_meep = f / light_speed
    sample_dt = 1.0 / (sample_oversample * f_meep)
    total_time = n_cycles / f_meep
    src_y = -0.25 * sy
    src = mp.Source(mp.GaussianSource(frequency=f_meep, fwidth=0.05*f_meep),
                    component=mp.Ez,
                    center=mp.Vector3(0, src_y, 0),
                    size=mp.Vector3(sx * 0.9, 0, 0))
    pml_layers = [mp.PML(thickness=0.5 * (light_speed / f))]
    cell = mp.Vector3(sx, sy, 0)
    sim = mp.Simulation(cell_size=cell,
                        geometry=geom,
                        boundary_layers=pml_layers,
                        sources=[src],
                        resolution=res,
                        Courant=cour)
    empty_thingy = []
    def collect(sim):
        try:
            empty_thingy.append(sim.get_field_point(mp.Ez, probe_point))
        except:
            empty_thingy.append(0.0)
    sim.run(mp.at_every(sample_dt, collect), until=total_time)
    return np.asarray(empty_thingy), sample_dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="./data")
    p.add_argument("--freq_GHz", default=0.3, type=float)
    p.add_argument("--n_width", default=12, type=int)
    p.add_argument("--n_bias", default=8, type=int)
    p.add_argument("--min_width_m", default=0.0005, type=float)
    p.add_argument("--max_width_m", default=0.005, type=float)
    p.add_argument("--min_eps", default=1.5, type=float)
    p.add_argument("--max_eps", default=6.0, type=float)
    p.add_argument("--cell_margin", default=0.002, type=float)
    p.add_argument("--resolution", default=200, type=int)
    p.add_argument("--probe_y_frac", default=0.30, type=float)
    p.add_argument("--thickness_m", default=0.001, type=float)
    p.add_argument("--sample_oversample", default=30, type=int)
    p.add_argument("--n_cycles", default=80, type=int)
    p.add_argument("--save_every", default=10, type=int)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    freq_hz = args.freq_GHz * 1e9
    lam = light_speed / freq_hz
    px_pw = args.resolution * lam
    if px_pw < 12:
        req_res = int(math.ceil(20.0 / lam))
        args.resolution = req_res

    widths = np.linspace(args.min_width_m, args.max_width_m, args.n_width)
    epsilons = np.linspace(args.min_eps, args.max_eps, args.n_bias)

    recs = []
    total = len(widths) * len(epsilons)
    cnt = 0
    t0 = time.time()

    for w in widths:
        for eps in epsilons:
            cnt += 1
            pml_thick = 0.5 * lam
            sx_guess = max(w + 2 * pml_thick + args.cell_margin, 3 * lam)
            sy_guess = max(args.thickness_m + 2 * pml_thick + args.cell_margin, 2 * lam)
            sx_adj, nx = ensure_integer_grid(sx_guess, args.resolution)
            sy_adj, ny = ensure_integer_grid(sy_guess, args.resolution)
            patch = mp.Block(center=mp.Vector3(0, 0, 0),
                             size=mp.Vector3(w, args.thickness_m, mp.inf),
                             material=mp.Medium(epsilon=eps))
            geom = [patch]
            probe_y = args.probe_y_frac * sy_adj
            probe_pt = mp.Vector3(0, probe_y, 0)
            try:
                ts_base, dt = run_time_series(freq_hz, [], sx_adj, sy_adj, args.resolution, probe_pt,
                                              sample_oversample=args.sample_oversample, n_cycles=args.n_cycles)
                ts_dev, dt2 = run_time_series(freq_hz, geom, sx_adj, sy_adj, args.resolution, probe_pt,
                                              sample_oversample=args.sample_oversample, n_cycles=args.n_cycles)
            except:
                continue
            ts_scatter = ts_dev - ts_base
            N = len(ts_scatter)
            if N < 8:
                continue
            window = np.hanning(N)
            fft_vals = np.fft.fft(ts_scatter * window)
            freqs_fft = np.fft.fftfreq(N, d=dt)
            target_f_meep =  freq_hz / light_speed
            idx = np.argmin(np.abs(freqs_fft - target_f_meep))
            comp = fft_vals[idx] / N
            mag = float(np.abs(comp))
            ph = float(np.angle(comp))
            recs.append({
                "width_m": float(w),
                "epsilon": float(eps),
                "freq_Hz": float(freq_hz),
                "sx_m": float(sx_adj),
                "sy_m": float(sy_adj),
                "nx": int(nx),
                "ny": int(ny),
                "pml_m": float(pml_thick),
                "refl_real": float(np.real(comp)),
                "refl_imag": float(np.imag(comp)),
                "refl_mag": mag,
                "refl_phase": ph
            })
            if (cnt % args.save_every) == 0:
                pd.DataFrame(recs).to_csv(Path(outdir)/"unitcell_dataset_partial.csv", index=False)
                meta = {"last_count": cnt, "time": time.time(), "args": vars(args)}
                with open(Path(outdir)/"metadata_partial.json","w") as f:
                    json.dump(meta, f, indent=2)

    df = pd.DataFrame(recs)
    csv_path = Path(outdir)/"unitcell_dataset.csv"
    df.to_csv(csv_path, index=False)
    print("Done", len(recs), "records in", time.time() - t0, "s")

if __name__ == "__main__":
    main()