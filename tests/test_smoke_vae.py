import torch


def test_vae_forward_and_loss_cpu():
    from generative_model_unlearning.generative_models import VAE, vae_loss_function

    device = torch.device("cpu")
    model = VAE(latent_dim=16).to(device)
    model.eval()

    # Create a small synthetic batch (batch_size, channels, H, W) matching CIFAR-like 32x32
    batch = torch.rand(4, 3, 32, 32, device=device)
    with torch.no_grad():
        recon, mu, logvar = model(batch)

    # Basic shape checks
    assert recon.shape == batch.shape
    assert mu.shape[0] == batch.shape[0]
    assert logvar.shape == mu.shape

    # Compute loss (should be a scalar)
    loss = vae_loss_function(recon, batch, mu, logvar)
    assert torch.is_tensor(loss) and loss.dim() == 0
