"""Deterministic held-out reconstruction evaluation; scores are not speaker similarity."""
import math


def record_validation(state, score, epoch, dataset):
    if not math.isfinite(score):
        raise ValueError('验证误差不是有限数，停止保存最佳模型')
    state = dict(state)
    baseline = state.setdefault('baseline', score)
    improved = score < state.get('best', float('inf'))
    if improved:
        state.update(best=score, best_epoch=epoch)
    state['dataset'] = dataset
    state['latest'] = dict(epoch=epoch, mel_mae=score,
                           improvement_percent=100*(baseline-score)/max(baseline, 1e-12),
                           best=state['best'], best_epoch=state['best_epoch'], improved=improved)
    return state, improved


def evaluate(generator, samples, cfg, device, speaker):
    import torch
    import torch.nn.functional as F
    from vits_fast_fine_tuning import commons
    from vits_fast_fine_tuning.mel_processing import spec_to_mel_torch, mel_spectrogram_torch
    data = cfg['data']
    fft, hop, rate = data['filter_length'], data['hop_length'], data['sampling_rate']
    was_training = generator.training
    try:
        generator.eval()
        with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []), torch.no_grad():
            torch.manual_seed(7341)
            scores = []
            for text, spec, wave in samples:
                x, spec = text.unsqueeze(0).to(device), spec.unsqueeze(0).to(device)
                output = generator(x, torch.tensor([x.shape[-1]], device=device), spec,
                                   torch.tensor([spec.shape[-1]], device=device),
                                   torch.tensor([speaker], device=device))
                generated, _, _, slices = output[:4]
                mel = spec_to_mel_torch(spec, fft, data['n_mel_channels'], rate,
                                        data['mel_fmin'], data.get('mel_fmax'))
                target = commons.slice_segments(mel, slices, cfg['train']['segment_size']//hop)
                predicted = mel_spectrogram_torch(generated.squeeze(1), fft, data['n_mel_channels'],
                                                  rate, hop, data['win_length'], data['mel_fmin'],
                                                  data.get('mel_fmax'))
                scores.append(F.l1_loss(predicted, target).item())
            return sum(scores)/len(scores)
    finally:
        generator.train(was_training)
