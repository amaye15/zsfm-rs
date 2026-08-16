mod chronos;
mod flowstate;
mod generic;
mod lag_llama;
mod mitra;
mod moirai;
mod moirai2;
mod moment;
mod sundial;
mod tabdpt;
mod tabfm;
mod tabicl;
mod tabpfn;
mod timesfm;
mod tirex;
mod toto;
mod ttm;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "zsfm", about = "Zero-shot foundation models — GGUF conversion + inference CLI")]
struct Cli {
    #[command(subcommand)]
    model: ModelCommand,
}

#[derive(Subcommand)]
enum ModelCommand {
    /// Toto-2 (Datadog) zero-shot time-series forecaster.
    Toto {
        #[command(subcommand)]
        command: toto::Command,
    },
    /// TabFM (google/tabfm-1.0.0-pytorch) zero-shot tabular classifier/regressor.
    Tabfm {
        #[command(subcommand)]
        command: tabfm::Command,
    },
    /// Mitra (autogluon/mitra-classifier, autogluon/mitra-regressor) zero-shot tabular
    /// foundation model.
    Mitra {
        #[command(subcommand)]
        command: mitra::Command,
    },
    /// TabDPT (Layer6/TabDPT) zero-shot tabular foundation model.
    Tabdpt {
        #[command(subcommand)]
        command: tabdpt::Command,
    },
    /// TabICL (jingang/TabICL, v2) zero-shot tabular foundation model (classification only).
    Tabicl {
        #[command(subcommand)]
        command: tabicl::Command,
    },
    /// TabPFN-3 (Prior-Labs/tabpfn_3) zero-shot tabular foundation model (classification only).
    /// NON-COMMERCIAL WEIGHTS LICENSE — see `zsfm tabpfn infer --help`.
    Tabpfn {
        #[command(subcommand)]
        command: tabpfn::Command,
    },
    /// Chronos-2 (Amazon) zero-shot time-series forecaster.
    Chronos {
        #[command(subcommand)]
        command: chronos::Command,
    },
    /// TimesFM 2.5 200M (Google) zero-shot time-series forecaster.
    Timesfm {
        #[command(subcommand)]
        command: timesfm::Command,
    },
    /// Sundial / Timer v3 (thuml) zero-shot time-series forecaster.
    Sundial {
        #[command(subcommand)]
        command: sundial::Command,
    },
    /// TinyTimeMixer / TTM-R2 (IBM Granite) zero-shot time-series forecaster.
    Ttm {
        #[command(subcommand)]
        command: ttm::Command,
    },
    /// Lag-Llama (ServiceNow) zero-shot time-series forecaster.
    LagLlama {
        #[command(subcommand)]
        command: lag_llama::Command,
    },
    /// MOMENT-1-large (CMU) zero-shot time-series forecaster.
    Moment {
        #[command(subcommand)]
        command: moment::Command,
    },
    /// Moirai-1.0-R-large (Salesforce) zero-shot time-series forecaster.
    Moirai {
        #[command(subcommand)]
        command: moirai::Command,
    },
    /// Moirai-2.0-R-small (Salesforce) zero-shot time-series forecaster.
    Moirai2 {
        #[command(subcommand)]
        command: moirai2::Command,
    },
    /// FlowState-R1 (IBM Granite) zero-shot time-series forecaster.
    Flowstate {
        #[command(subcommand)]
        command: flowstate::Command,
    },
    /// TiRex (NX-AI) zero-shot time-series forecaster.
    Tirex {
        #[command(subcommand)]
        command: tirex::Command,
    },
    /// Convert ANY checkpoint (safetensors / PyTorch pickle / npy / npz / ONNX /
    /// HDF5 / Keras / GGUF) to GGUF, tensor names passed through. GGUF input
    /// re-quantizes.
    Convert(generic::ConvertArgs),
    /// List tensors (and optionally values) of any supported checkpoint format.
    Inspect(generic::InspectArgs),
    /// Download a HuggingFace repo's checkpoint in whatever format it's
    /// published in (safetensors / pickle / npy / npz / ONNX / HDF5 / Keras /
    /// GGUF), without converting.
    Pull(generic::PullArgs),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.model {
        ModelCommand::Toto { command } => toto::run(command).await,
        ModelCommand::Tabfm { command } => tabfm::run(command).await,
        ModelCommand::Mitra { command } => mitra::run(command).await,
        ModelCommand::Tabdpt { command } => tabdpt::run(command).await,
        ModelCommand::Tabicl { command } => tabicl::run(command).await,
        ModelCommand::Tabpfn { command } => tabpfn::run(command).await,
        ModelCommand::Chronos { command } => chronos::run(command).await,
        ModelCommand::Timesfm { command } => timesfm::run(command).await,
        ModelCommand::Sundial { command } => sundial::run(command).await,
        ModelCommand::Ttm { command } => ttm::run(command).await,
        ModelCommand::LagLlama { command } => lag_llama::run(command).await,
        ModelCommand::Moment { command } => moment::run(command).await,
        ModelCommand::Moirai { command } => moirai::run(command).await,
        ModelCommand::Moirai2 { command } => moirai2::run(command).await,
        ModelCommand::Flowstate { command } => flowstate::run(command).await,
        ModelCommand::Tirex { command } => tirex::run(command).await,
        ModelCommand::Convert(args) => generic::run_convert(args).await,
        ModelCommand::Inspect(args) => generic::run_inspect(args),
        ModelCommand::Pull(args) => generic::run_pull(args).await,
    }
}
