use std::path::Path;

use ort::{ep, session::Session};

use crate::Result;

/// Where ONNX sessions execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// ONNX Runtime's CPU execution provider.
    #[default]
    Cpu,
    /// DirectML: runs on any DirectX 12 device. Falls back to the CPU
    /// provider if registration fails.
    DirectMl,
}

/// Graph rewriting effort applied while building a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GraphOptimization {
    /// Keep the model graph untouched.
    Disabled,
    /// Constant folding and simple rewrites (ORT `Level1`).
    #[default]
    Basic,
    /// Extended fusions such as convolution + activation (ORT `Level2`).
    Extended,
    /// Everything ORT offers, including full graph fusion (ORT `Level3`).
    All,
}

impl GraphOptimization {
    fn to_ort(self) -> ort::session::builder::GraphOptimizationLevel {
        use ort::session::builder::GraphOptimizationLevel as Level;
        match self {
            Self::Disabled => Level::Disable,
            Self::Basic => Level::Level1,
            Self::Extended => Level::Level2,
            Self::All => Level::All,
        }
    }
}

/// How ONNX sessions are built for both models.
///
/// Settings a backend ignores (for example `device_id` on the CPU backend)
/// are silently dropped.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SessionOptions {
    /// Execution backend.
    pub backend: Backend,
    /// DirectML adapter index; `None` selects the default adapter.
    pub device_id: Option<i32>,
    /// Graph optimization effort applied before execution.
    pub graph_optimization: GraphOptimization,
    /// Intra-op thread count; `None` uses every available core.
    pub intra_threads: Option<usize>,
    /// Ask DirectML for deterministic compute (slower, an accuracy knob).
    pub deterministic: bool,
}

impl SessionOptions {
    /// Options for CPU-only sessions.
    pub fn cpu() -> Self {
        Self {
            backend: Backend::Cpu,
            ..Self::default()
        }
    }

    /// Options for DirectML sessions.
    pub fn directml() -> Self {
        Self {
            backend: Backend::DirectMl,
            ..Self::default()
        }
    }

    /// Select a specific DirectML adapter.
    pub fn with_device_id(mut self, device_id: i32) -> Self {
        self.device_id = Some(device_id);
        self
    }

    /// Override the graph optimization effort.
    pub fn with_graph_optimization(mut self, level: GraphOptimization) -> Self {
        self.graph_optimization = level;
        self
    }

    /// Set the intra-op thread count instead of using every available core.
    pub fn with_intra_threads(mut self, threads: usize) -> Self {
        self.intra_threads = Some(threads);
        self
    }

    /// Toggle DirectML's deterministic compute mode.
    pub fn with_deterministic(mut self, deterministic: bool) -> Self {
        self.deterministic = deterministic;
        self
    }
}

/// Build a session for `model`.
///
/// ORT falls back to CPU per operation when the DirectML execution provider
/// cannot handle a node (DequantizeLinear and friends), so provider
/// registration failures degrade gracefully instead of erroring.
pub(crate) fn open_session(model: &Path, options: &SessionOptions) -> Result<Session> {
    let builder = Session::builder()?;

    // Builder errors carry the session builder itself (and are not Send/Sync),
    // so they are recovered from rather than propagated.
    let builder = builder
        .with_optimization_level(options.graph_optimization.to_ort())
        .unwrap_or_else(|e| e.recover());

    let threads = options
        .intra_threads
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()));
    let builder = match threads {
        Some(threads) => builder
            .with_intra_threads(threads)
            .unwrap_or_else(|e| e.recover()),
        None => builder,
    };

    let builder = if options.deterministic {
        builder
            .with_deterministic_compute(true)
            .unwrap_or_else(|e| e.recover())
    } else {
        builder
    };

    let mut builder = match options.backend {
        Backend::Cpu => builder,
        Backend::DirectMl => {
            // The DML graph fusion transformer mis-compiles the cleaner's mask
            // branch (mask logits attenuated ~1.56x with a ~+1.5 bias, in both
            // fp32 and int8 graphs). With fusion disabled the DML output is
            // bit-exact with the CPU session (mean abs diff ~1e-7) at ~10%
            // slower pace; metacommands, optimization level and deterministic
            // compute were all ruled out as causes.
            let builder = builder
                .with_config_entry("ep.dml.disable_graph_fusion", "1")
                .unwrap_or_else(|e| e.recover());

            let provider = match options.device_id {
                Some(device_id) => ep::DirectML::default().with_device_id(device_id),
                None => ep::DirectML::default(),
            };
            builder
                .with_execution_providers([provider.build()])
                .unwrap_or_else(|e| e.recover())
        }
    };

    Ok(builder.commit_from_file(model)?)
}
