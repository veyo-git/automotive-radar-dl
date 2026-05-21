"""Gradio interactive dashboard for automotive radar interference demo.

Three tabs:
    1. Simulator: Tune radar parameters, generate RD maps with interference.
    2. Classifier: Real-time interference type classification.
    3. Denoiser: Side-by-side interference mitigation demo.
"""
import os
import sys
import io
import numpy as np
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.simulator.waveform import generate_chirp, generate_if_signal, adc_sampling
from src.simulator.processing import range_doppler_map, compute_axes
from src.simulator.interference import (
    generate_cw_interference,
    generate_cross_fmcw_interference,
    generate_noise_jamming,
    generate_spoofing,
    generate_chirp_swept_interference,
    inject_interference,
    INTERFERENCE_LABELS,
)
from src.training.config import RadarConfig
from src.viz.plots import (
    plot_rd_map,
    plot_interference_comparison,
    figure_to_bytes,
)


# ---------------------------------------------------------------------------
# Model loading (lazy)
# ---------------------------------------------------------------------------

_classifier_model = None
_denoiser_model = None
_device = None
_radar = RadarConfig()


def get_device():
    global _device
    if _device is None:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def load_classifier(checkpoint_path="models/classifier.pth"):
    global _classifier_model
    if _classifier_model is not None:
        return _classifier_model
    import torch
    from src.models.classifier import InterferenceClassifier
    if not os.path.exists(checkpoint_path):
        return None
    ckpt = torch.load(checkpoint_path, map_location=get_device(), weights_only=False)
    n_classes = len(ckpt.get("class_names", INTERFERENCE_LABELS))
    model = InterferenceClassifier(n_classes=n_classes).to(get_device())
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _classifier_model = model
    return model


def load_denoiser(checkpoint_path="models/denoiser.pth"):
    global _denoiser_model
    if _denoiser_model is not None:
        return _denoiser_model
    import torch
    from src.models.denoiser import RadarUNet
    if not os.path.exists(checkpoint_path):
        return None
    ckpt = torch.load(checkpoint_path, map_location=get_device(), weights_only=False)
    features = ckpt.get("features", (64, 128, 256, 512))
    model = RadarUNet(features=features).to(get_device())
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _denoiser_model = model
    return model


# ---------------------------------------------------------------------------
# Simulator tab
# ---------------------------------------------------------------------------

def run_simulator(
    target_range, target_velocity, target_rcs,
    interference_type, inr_db, snr_db,
    fc_ghz, bandwidth_mhz, chirp_duration_us, n_chirps,
):
    """Generate clean + interfered RD maps based on user parameters."""
    tgt_range = float(target_range)
    tgt_velocity = float(target_velocity)
    tgt_rcs = float(target_rcs)

    fc = fc_ghz * 1e9
    B = bandwidth_mhz * 1e6
    T_chirp = chirp_duration_us * 1e-6
    N_chirps = int(n_chirps)
    fs = _radar.fs
    N_samp = _radar.N_samp

    # Generate signals
    chirp, t = generate_chirp(fc, B, T_chirp, fs)

    targets = [{"range": tgt_range, "velocity": tgt_velocity, "rcs": tgt_rcs}]
    s_if, tgt_params = generate_if_signal(t, targets, fc, B, T_chirp)

    # Add thermal noise
    snr_linear = 10 ** (snr_db / 10)
    signal_power = np.mean(np.abs(s_if) ** 2) + 1e-12
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
    s_clean = s_if + noise

    # Generate interference
    int_type_name = interference_type
    if int_type_name == "clean":
        s_interfered = s_clean
        label = 0
    else:
        label_map = {"cw": 1, "cross_fmcw": 2, "noise_jamming": 3,
                     "spoofing": 4, "chirp_swept": 5}
        label = label_map.get(int_type_name, 1)

        K_victim = B / T_chirp
        if label == 1:  # CW
            i_sig = generate_cw_interference(t, fc_if=2e6, amplitude=1.0)
        elif label == 2:  # Cross-FMCW
            K_interf = K_victim * 1.3
            i_sig = generate_cross_fmcw_interference(
                t, K_victim, K_interf, delta_t=0.3e-6, fc_offset=1e6
            )
        elif label == 3:  # Noise
            i_sig = generate_noise_jamming(t, amplitude=1.0)
        elif label == 4:  # Spoofing
            i_sig = generate_spoofing(t, fc, B, T_chirp,
                                       fake_range=80, fake_velocity=-8)
        elif label == 5:  # Chirp-swept
            i_sig = generate_chirp_swept_interference(
                t, f_start=0.5e6, f_stop=4e6
            )
        else:
            i_sig = np.zeros_like(t)

        s_interfered, _ = inject_interference(
            s_clean, i_sig, snr_db=snr_db, inr_db=inr_db
        )

    # ADC -> RD processing
    adc_clean_np = adc_sampling(s_clean, N_samp, N_chirps)
    adc_int_np = adc_sampling(s_interfered, N_samp, N_chirps)

    # Per-chirp Doppler
    f_d = 2 * tgt_velocity * fc / 299792458.0
    for chirp_idx in range(N_chirps):
        phase = 2 * np.pi * f_d * chirp_idx * T_chirp
        adc_clean_np[chirp_idx] *= np.exp(1j * phase)
        adc_int_np[chirp_idx] *= np.exp(1j * phase)

    rd_clean_db, _ = range_doppler_map(adc_clean_np)
    rd_int_db, _ = range_doppler_map(adc_int_np)
    range_bins, vel_bins = compute_axes(fc, B, T_chirp, N_chirps, N_samp)

    # Generate plots
    fig_clean = plot_rd_map(rd_clean_db, range_bins, vel_bins,
                             title="Clean RD Map")[0]
    fig_int = plot_rd_map(rd_int_db, range_bins, vel_bins,
                           title=f"Interfered ({int_type_name}, INR={inr_db} dB)")[0]

    buf_clean = figure_to_bytes(fig_clean)
    buf_int = figure_to_bytes(fig_int)

    # Store for classifier/denoiser
    global _last_rd
    _last_rd = {
        "clean": rd_clean_db,
        "interfered": rd_int_db,
        "label": label,
        "range_bins": range_bins,
        "vel_bins": vel_bins,
    }

    return buf_clean, buf_int, tgt_params[0]["f_beat"]


# ---------------------------------------------------------------------------
# Classifier tab
# ---------------------------------------------------------------------------

def run_classification():
    """Classify the last generated RD map."""
    import torch
    if "_last_rd" not in dir() or _last_rd is None:
        return "Run the Simulator first to generate an RD map.", None

    model = load_classifier()
    if model is None:
        return "No classifier checkpoint found. Train the model first.", None

    rd = _last_rd["interfered"]
    # Normalize
    rd_norm = (rd - rd.mean()) / (rd.std() + 1e-8)
    x = torch.from_numpy(rd_norm).float().unsqueeze(0).unsqueeze(0).to(get_device())

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    result = "**Interference Classification Results**\n\n"
    result += f"| Type | Probability |\n|------|------------|\n"
    for i, (name, prob) in enumerate(zip(INTERFERENCE_LABELS.values(), probs)):
        bar = "█" * int(prob * 30)
        result += f"| {name} | {prob:.3f} {bar} |\n"

    pred = int(torch.argmax(logits, dim=1).item())
    result += f"\n**Predicted:** {INTERFERENCE_LABELS[pred]} (confidence: {probs[pred]:.1%})"
    result += f"\n**True label:** {INTERFERENCE_LABELS[_last_rd['label']]}"

    return result, None


# ---------------------------------------------------------------------------
# Denoiser tab
# ---------------------------------------------------------------------------

def run_denoising():
    """Denoise the last generated RD map."""
    import torch
    if "_last_rd" not in dir() or _last_rd is None:
        return "Run the Simulator first to generate an RD map.", None

    model = load_denoiser()
    if model is None:
        return "No denoiser checkpoint found. Train the model first.", None

    clean = _last_rd["clean"]
    interfered = _last_rd["interfered"]
    range_bins = _last_rd["range_bins"]
    vel_bins = _last_rd["vel_bins"]

    # Normalize
    mean_int = interfered.mean()
    std_int = interfered.std() + 1e-8
    int_norm = (interfered - mean_int) / std_int
    clean_norm = (clean - mean_int) / std_int

    x = torch.from_numpy(int_norm).float().unsqueeze(0).unsqueeze(0).to(get_device())
    with torch.no_grad():
        output = model(x)
        denoised_norm = output[0, 0].cpu().numpy()
        # Denormalize
        denoised = denoised_norm * std_int + mean_int

    fig_comp = plot_interference_comparison(
        clean, interfered, denoised,
        range_bins=range_bins, velocity_bins=vel_bins,
        titles=["Clean", "Interfered", "Denoised (U-Net)"]
    )

    buf = figure_to_bytes(fig_comp)

    # Metrics
    int_error = np.mean((interfered - clean) ** 2)
    den_error = np.mean((denoised - clean) ** 2)
    if int_error > 1e-10:
        sinr_imp = 10 * np.log10(int_error / den_error)
    else:
        sinr_imp = 0.0

    result = f"**Denoising Results**\n\n"
    result += f"- SINR Improvement: **{sinr_imp:.1f} dB**\n"
    result += f"- MSE (interfered): {int_error:.4f}\n"
    result += f"- MSE (denoised): {den_error:.4f}\n"

    return result, buf


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------

def create_app():
    """Create and return the Gradio Blocks app."""
    css = """
    .gradio-container { max-width: 1100px !important; }
    .result-text { font-family: monospace; font-size: 14px; }
    """

    with gr.Blocks(title="Automotive Radar DL Demo", css=css, theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # Automotive Radar Signal Simulation & DL-Based Interference Mitigation

            **FMCW Radar Simulator + CNN Interference Classifier + U-Net Denoiser**

            Simulate 77 GHz automotive FMCW radar, inject realistic interference,
            classify interference types with a CNN, and recover clean signals
            with a U-Net denoiser.
            """
        )

        # ===== Tab 1: Simulator =====
        with gr.Tab("1. Simulator"):
            gr.Markdown("### FMCW Radar Simulator with Interference Injection")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Target Parameters")
                    target_range = gr.Slider(5, 200, value=50, step=1, label="Range (m)")
                    target_velocity = gr.Slider(-30, 30, value=10, step=1, label="Velocity (m/s)")
                    target_rcs = gr.Slider(0.1, 100, value=10, step=0.1, label="RCS (m²)")

                    gr.Markdown("#### Interference")
                    interference_type = gr.Dropdown(
                        choices=["clean", "cw", "cross_fmcw", "noise_jamming",
                                 "spoofing", "chirp_swept"],
                        value="cross_fmcw", label="Interference Type"
                    )
                    inr_db = gr.Slider(-10, 30, value=15, step=1, label="INR (dB)")
                    snr_db = gr.Slider(5, 40, value=25, step=1, label="SNR (dB)")

                    gr.Markdown("#### Radar Parameters")
                    fc_ghz = gr.Slider(76, 81, value=77, step=0.1, label="Carrier Freq (GHz)")
                    bandwidth_mhz = gr.Slider(100, 2000, value=750, step=50, label="Bandwidth (MHz)")
                    chirp_duration_us = gr.Slider(10, 100, value=40, step=5, label="Chirp Duration (µs)")
                    n_chirps = gr.Slider(16, 256, value=64, step=16, label="Chirps per Frame")

                    run_btn = gr.Button("Generate RD Maps", variant="primary")

                with gr.Column(scale=1):
                    with gr.Row():
                        clean_plot = gr.Image(label="Clean RD Map", type="filepath")
                        int_plot = gr.Image(label="Interfered RD Map", type="filepath")
                    beat_freq = gr.Number(label="Beat Frequency (Hz)", precision=0)

            run_btn.click(
                fn=run_simulator,
                inputs=[target_range, target_velocity, target_rcs,
                        interference_type, inr_db, snr_db,
                        fc_ghz, bandwidth_mhz, chirp_duration_us, n_chirps],
                outputs=[clean_plot, int_plot, beat_freq],
            )

        # ===== Tab 2: Classifier =====
        with gr.Tab("2. Classifier"):
            gr.Markdown("### CNN Interference Type Classification")
            gr.Markdown("Classify the interference type from the RD map generated in the Simulator tab.")

            cls_btn = gr.Button("Run Classification", variant="primary")
            cls_result = gr.Markdown("Run the Simulator first, then click Classify.", elem_classes="result-text")

            cls_btn.click(fn=run_classification, inputs=[], outputs=[cls_result])

        # ===== Tab 3: Denoiser =====
        with gr.Tab("3. Denoiser"):
            gr.Markdown("### U-Net Interference Mitigation")
            gr.Markdown("Recover the clean RD map from the interfered version using a trained U-Net.")

            den_btn = gr.Button("Run Denoising", variant="primary")
            den_result = gr.Markdown("Run the Simulator first, then click Denoise.", elem_classes="result-text")
            den_plot = gr.Image(label="Clean / Interfered / Denoised Comparison", type="filepath")

            den_btn.click(fn=run_denoising, inputs=[], outputs=[den_result, den_plot])

        # ===== Footer =====
        gr.Markdown(
            """
            ---
            **Tech Stack:** Python · PyTorch · NumPy · Gradio · FMCW Radar Physics

            [GitHub Repository](https://github.com) | [Documentation](docs/)
            """
        )

    return app


# Global state
_last_rd = None


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)
