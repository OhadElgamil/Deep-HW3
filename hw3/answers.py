r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 64
    hypers['seq_len'] = 128
    hypers['h_dim'] = 128
    hypers['n_layers'] = 2
    hypers['dropout'] = 2e-1
    hypers['learn_rate'] = 1e-3
    hypers['lr_sched_factor'] = 5e-1
    hypers['lr_sched_patience'] = 3
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "To be, or not to be, that is the question.\n"
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**
"""

part1_q2 = r"""
**Your answer:**
"""

part1_q3 = r"""
**Your answer:**
"""

part1_q4 = r"""
**Your answer:**
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 8
    hypers['h_dim'] = 512
    hypers['z_dim'] = 16
    hypers['x_sigma2'] = 0.003
    hypers['learn_rate'] = 0.0002
    hypers['betas'] = (0.5, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
The hyperparameter $\sigma^2_x$ represents the assumed variance of the observation noise in the probabilistic decoder. In practical terms, it controls the between the reconstruction accuracy and the latent space regularization.

By tuning $\sigma^2_x$, we directly affect the penalty for reconstruction errors:

* **High Value:**
    A high variance implies we "trust" the pixel values less. This scales down the reconstruction loss, causing the model to prioritize the KL-divergence term (regularization).
    Therefore, the model produces blurry images because it is not strictly penalized for losing high-frequency details, preferring a smooth latent space instead.

* **Low Value:**
    A low variance implies we demand high precision. This scales up the reconstruction loss, forcing the model to prioritize pixel-perfect accuracy over the KL prior.
    Therefore, the model produces sharp, detailed images. However, setting this too low can lead to overfitting or an unstable latent space (approaching a deterministic Autoencoder).
"""

part2_q2 = r"""
The VAE loss function combines two conflicting goals:

**1. Reconstruction Loss**
* Purpose: Measures the pixel-wise difference between the input image $x$ and the reconstructed image $x'$.
* Role: Forces the Decoder to learn how to rebuild the image from the compressed latent vector $z$. Without this, the output would just be noise.

**2. KL Divergence Loss**
* Purpose: Measures the difference between our learned latent distribution $q(z|x)$ and a standard Gaussian prior $p(z) = \mathcal{N}(0, I)$.
* Role: Acts as a regularizer. It prevents the Encoder from "cheating" by mapping images to far-away, isolated points. It forces the latent variables to stay close to 0 and follow a known distribution.

---

**Effect on Latent Space**
The KL term forces the latent space to be continuous and dense.
Instead of memorizing data points as isolated spikes (like standard Autoencoders), the VAE is forced to map inputs to overlapping probability clouds centered at the origin.

**Benefit**
The main benefit is generative capability:
1.  Valid Sampling: Since we forced the space to look like a standard Normal distribution, we can simply pick a random vector from $\mathcal{N}(0, I)$, feed it to the decoder, and get a valid image.
2.  Smooth Interpolation: Because the space is continuous, moving slightly in the latent space results in smooth changes in the image rather than sudden jumps.
"""

part2_q3 = r"""
We start by maximizing the evidence distribution $p_{\theta}(\mathbf{x})$ because this corresponds to the principle of Maximum Likelihood Estimation (MLE), which aims to find the model parameters that maximize the 
probability of observing our actual training data. However, directly computing this evidence is intractable because it requires marginalizing over the latent variables, $p_{\theta}(\mathbf{x}) = \int p_{\theta}(\mathbf{x}|\mathbf{z})p(\mathbf{z})d\mathbf{z}$, 
which involves an impossible integral for complex neural networks. Therefore, since we cannot maximize the evidence directly, we derive and maximize a tractable lower bound known as ELBO. Optimizing this lower bound guarantees that we are also increasing the 
true log-likelihood of the data.
"""

part2_q4 = r"""
We model the log-variance $\log(\sigma^2_{\alpha})$ instead of the variance directly to ensure numerical stability and simplify the optimization process. Since a variance must always be strictly non-negative ($\sigma^2 \ge 0$), directly predicting it would 
require constraining the network's output to prevent invalid negative values. By predicting the logarithm, the network can output values in the entire real number range $(-\infty, \infty)$. Exponentiating this output guarantees a positive variance, allowing 
the optimizer to learn unconstrained weights without worrying about boundary conditions.
"""


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    hypers['embed_dim'] = 128
    hypers['num_heads'] = 8
    hypers['num_layers'] = 2
    hypers['hidden_dim'] = 256
    hypers['window_size'] = 64
    hypers['droupout'] = 0.2
    hypers['lr'] = 3e-4
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
"""

part3_q2 = r"""
**Your answer:**
"""

# ==============
